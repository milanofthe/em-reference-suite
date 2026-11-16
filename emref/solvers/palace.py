"""Palace reference of a case: one gmsh tetrahedral mesh of emref.model3d, solved by
Palace at element order 2 and 3, written as reference.sNp (order 3) and meta.yaml.

python -m emref reference <case> palace --cores N [--out DIR]

Conductors are volumes with their bulk conductivity, meshed inside. Stack boundaries
of kind conductor are Palace Conductivity boundaries (a conducting half space), pec
is PEC, open faces are absorbing. Every port is excited in turn (full S matrix).
Needs gmsh (Python) and the palace launcher on PATH; scratch files go to TMPDIR.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml
from shapely import Point, Polygon
from shapely.ops import unary_union

from .. import model3d
from ..case import CONVERGED_BOUND, REFERENCE_DIR, Case, max_ds

# --- parameters, all written into meta.yaml ---------------------------------------
# Box: D = max(height of the structure, its smaller lateral extent).
BOX_MARGIN = 0.75           # lateral margin beyond the structure, in D
BOX_AIR = 1.5               # air above the structure (open top), in D
# Mesh size: on a conductor its narrowest drawn width or gap / ACROSS, growing by
# GROWTH per unit distance from it, capped at the wavelength at f_max in the local
# material / PER_WAVELENGTH.
ACROSS = 4
GROWTH = 0.5
PER_WAVELENGTH = 10
FEATURE_TOL = 1e-3          # area fraction an opening / closing must change to find a width or gap
ORDERS = (2, 3)
EXTRA_ORDER = 4             # solved on the same mesh when ORDERS end above the bound and it fits in memory
ABSORBING_ORDER = 2
ADAPTIVE_TOL = 1e-3
ADAPTIVE_MAX_SAMPLES = 40
FREQ_POINTS = 201
LINEAR = {"Type": "MUMPS", "KSPType": "GMRES", "Tol": 1e-8, "MaxIts": 500,
          "ComplexCoarseSolve": True, "EstimatorTol": 1e-3}

C0 = 299792458.0
UM = 1e-6
PROGRESS = re.compile(r"global unknowns|ND \(p|Greedy iteration|Adaptive sampling|"
                      r"Total offline|did not converge|rror")


def _polygon(s: model3d.Solid) -> Polygon:
    return Polygon(s.outer, s.holes)


def narrowest(region) -> float:
    """Narrowest drawn width or gap of a region: twice the smallest radius at which a
    morphological opening (width) or closing (gap) changes its area by FEATURE_TOL."""
    area = region.area

    def changed(r: float) -> bool:
        opened = region.buffer(-r, join_style="mitre").buffer(r, join_style="mitre")
        closed = region.buffer(r, join_style="mitre").buffer(-r, join_style="mitre")
        return max(opened.symmetric_difference(region).area,
                   closed.symmetric_difference(region).area) > FEATURE_TOL * area

    lo, hi = 0.0, math.sqrt(area)
    while hi - lo > 1e-3 * hi:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if changed(mid) else (mid, hi)
    return 2 * hi


def box(model: model3d.Model) -> dict:
    (x0, y0, z0), (x1, y1, z1) = model.bbox
    d = max(z1 - z0, min(x1 - x0, y1 - y0))
    m = BOX_MARGIN * d
    top = model.boundaries["top"]
    zt = max(z1, top.z) + BOX_AIR * d if top.kind == "open" else top.z
    return {"x": (x0 - m, x1 + m), "y": (y0 - m, y1 + m),
            "z": (model.boundaries["bottom"].z, zt)}


def _conductors(model: model3d.Model) -> list[dict]:
    """Per conductor: its footprint, z range and mesh size on its surface."""
    out = {}
    for s in model.solids:
        if s.role in ("conductor", "via"):
            out.setdefault(s.name, {"name": s.name, "z": (s.z0, s.z1), "parts": []})["parts"].append(
                _polygon(s))
    for c in out.values():
        c["region"] = unary_union(c["parts"])
        c["narrowest_um"] = narrowest(c["region"])
        c["size_um"] = c["narrowest_um"] / ACROSS
        del c["parts"]
    return list(out.values())


def mesh(model: model3d.Model, path: Path) -> dict:
    """Mesh the model into path (msh 2.2); returns the attribute map and mesh sizes."""
    import gmsh
    for m in model.materials.values():
        if m.kind == "conductor" and m.pec:
            raise ValueError(f"{model.case}: PEC conductor {m.name} is not supported")
    b = box(model)
    (bx0, bx1), (by0, by1), (bz0, bz1) = b["x"], b["y"], b["z"]
    f_max = model.band_hz[1]

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        occ = gmsh.model.occ
        inputs = [((3, occ.addBox(bx0, by0, bz0, bx1 - bx0, by1 - by0, bz1 - bz0)), None, -1)]
        for s in model.solids:
            if s.outer is None:
                tag = occ.addBox(bx0, by0, s.z0, bx1 - bx0, by1 - by0, s.z1 - s.z0)
            else:
                loops = []
                for ring in (s.outer, *s.holes):
                    pts = [occ.addPoint(x, y, s.z0) for x, y in ring]
                    lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
                    loops.append(occ.addCurveLoop(lines))
                face = occ.addPlaneSurface(loops)
                tag = [t for d, t in occ.extrude([(2, face)], 0, 0, s.z1 - s.z0) if d == 3][0]
            inputs.append(((3, tag), s.material, s.priority))
        for p in model.ports:
            pts = [occ.addPoint(*c) for c in p.corners]
            lines = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
            inputs.append(((2, occ.addPlaneSurface([occ.addCurveLoop(lines)])), p.name, None))
        _, children = occ.fragment([inputs[0][0]], [i[0] for i in inputs[1:]])
        occ.synchronize()

        owner: dict[int, tuple[int, str | None]] = {}
        port_faces: dict[str, list[int]] = {}
        for (dimtag, name, prio), kids in zip(inputs, children):
            for d, t in kids:
                if prio is None and d == 2:
                    port_faces.setdefault(name, []).append(t)
                elif d == 3 and (t not in owner or prio > owner[t][0]):
                    owner[t] = (prio, name)
        groups: dict[str | None, list[int]] = {}
        for t, (_, name) in sorted(owner.items()):
            groups.setdefault(name, []).append(t)
        attr = {name: k + 1 for k, name in enumerate(groups)}
        for name, tags in groups.items():
            gmsh.model.addPhysicalGroup(3, tags, attr[name], name=name or "vacuum")

        # exterior faces: absorbing, or the stack boundary where it borders the stack
        bnd: dict[str, list[int]] = {"absorbing": [], "bottom": [], "top": []}
        eps = 1e-6 * max(bx1 - bx0, by1 - by0, bz1 - bz0)
        for _, f in gmsh.model.getEntities(2):
            vols = gmsh.model.getAdjacencies(2, f)[0]
            if len(vols) != 1:
                continue
            fz0, fz1 = gmsh.model.getBoundingBox(2, f)[2::3]
            side = "bottom" if fz1 < bz0 + eps else "top" if fz0 > bz1 - eps else None
            kind = model.boundaries[side].kind if side else "open"
            inside = owner[int(vols[0])][1] is not None
            bnd[side if kind != "open" and inside else "absorbing"].append(f)
        nxt = len(attr) + 1
        battr = {}
        for k, tags in bnd.items():
            if tags:
                gmsh.model.addPhysicalGroup(2, tags, nxt, name=k)
                battr[k] = nxt
                nxt += 1
        pattr = {}
        for p in model.ports:
            gmsh.model.addPhysicalGroup(2, port_faces[p.name], nxt, name=p.name)
            pattr[p.name] = nxt
            nxt += 1

        # mesh size: per entity the wavelength cap of its material, then the conductors
        def cap(name):
            m = model.materials.get(name)
            er = 1.0 if m is None or m.kind == "conductor" else m.er
            return C0 / (f_max * math.sqrt(er)) / UM / PER_WAVELENGTH
        caps = {(3, t): cap(name) for t, (_, name) in owner.items()}
        for d in (2, 1, 0):
            for _, t in gmsh.model.getEntities(d):
                up = gmsh.model.getAdjacencies(d, t)[0]
                caps[(d, t)] = min((caps[(d + 1, int(u))] for u in up), default=cap(None))
        conductors = _conductors(model)
        for c in conductors:
            c["bounds"] = c["region"].bounds

        def size(dim, tag, x, y, z, lc):
            s = caps.get((dim, tag), cap(None))
            for c in conductors:
                h = c["size_um"]
                (cx0, cy0, cx1, cy1), (cz0, cz1) = c["bounds"], c["z"]
                dz = max(cz0 - z, z - cz1, 0.0)
                lower = math.hypot(max(cx0 - x, x - cx1, 0.0), max(cy0 - y, y - cy1, 0.0), dz)
                if h + GROWTH * lower >= s:
                    continue
                s = min(s, h + GROWTH * math.hypot(c["region"].distance(Point(x, y)), dz))
            return s

        gmsh.model.mesh.setSizeCallback(size)
        for k in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
            gmsh.option.setNumber(f"Mesh.{k}", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 1)
        gmsh.write(str(path))
        tets = int(sum(len(t) for t in gmsh.model.mesh.getElementsByType(4)[:1]))
    finally:
        gmsh.finalize()
    return {"volumes": attr, "boundaries": battr, "ports": pattr, "tets": tets, "box": b,
            "conductors": {c["name"]: {"narrowest_um": round(c["narrowest_um"], 4),
                                       "size_um": round(c["size_um"], 4)} for c in conductors}}


def config(model: model3d.Model, m: dict, mesh_path: Path, order: int, out: Path) -> dict:
    mats = []
    for name, a in m["volumes"].items():
        mt = model.materials.get(name)
        e = {"Attributes": [a]}
        if mt is None:
            e["Permittivity"] = 1.0
        elif mt.kind == "conductor":
            e["Conductivity"] = mt.sigma
        else:
            e["Permittivity"] = mt.er
            if mt.tand:
                e["LossTan"] = mt.tand
            if mt.sigma:
                e["Conductivity"] = mt.sigma
        mats.append(e)
    bnd: dict = {"Absorbing": {"Attributes": [m["boundaries"]["absorbing"]], "Order": ABSORBING_ORDER}}
    for side in ("bottom", "top"):
        if side in m["boundaries"]:
            b = model.boundaries[side]
            a = [m["boundaries"][side]]
            if b.kind == "pec":
                bnd.setdefault("PEC", {"Attributes": []})["Attributes"] += a
            else:
                bnd.setdefault("Conductivity", []).append({"Attributes": a, "Conductivity": b.sigma})
    bnd["LumpedPort"] = [{"Index": k, "Attributes": [m["ports"][p.name]], "Direction": list(p.direction),
                          "R": p.z0_ohm, "Excitation": k} for k, p in enumerate(model.ports, 1)]
    f0, f1 = (f / 1e9 for f in model.band_hz)
    return {
        "Problem": {"Type": "Driven", "Verbose": 2, "Output": str(out)},
        "Model": {"Mesh": str(mesh_path), "L0": UM},
        "Domains": {"Materials": mats},
        "Boundaries": bnd,
        "Solver": {"Order": order, "Device": "CPU", "Linear": LINEAR,
                   "Driven": {"Samples": [{"Type": "Linear", "MinFreq": f0, "MaxFreq": f1,
                                           "NSample": FREQ_POINTS}],
                              "AdaptiveTol": ADAPTIVE_TOL,
                              "AdaptiveMaxSamples": ADAPTIVE_MAX_SAMPLES}},
    }


def read_s(postpro: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    """port-S.csv -> f [Hz], S [f, n, n]."""
    rows = [r for r in csv.reader(open(postpro / "port-S.csv")) if r]
    head = [h.strip() for h in rows[0]]
    data = np.array([[float(x) for x in r] for r in rows[1:]])
    f = np.unique(data[:, 0])
    s = np.full((len(f), n, n), np.nan, complex)
    for k, h in enumerate(head):
        g = re.fullmatch(r"\|S\[(\d+)\]\[(\d+)\]\| \(dB\)", h)
        if not g:
            continue
        i, j = int(g[1]) - 1, int(g[2]) - 1
        ph = head.index(f"arg(S[{i + 1}][{j + 1}]) (deg.)")
        for row in data:
            if not np.isnan(row[k]):
                s[np.searchsorted(f, row[0]), i, j] = 10 ** (row[k] / 20) * np.exp(1j * np.deg2rad(row[ph]))
    if np.isnan(s).any():
        raise RuntimeError(f"{postpro}/port-S.csv: S matrix incomplete")
    return f * 1e9, s


def _log_value(log: str, pattern: str) -> str:
    g = re.search(pattern, log)
    if not g:
        raise RuntimeError(f"palace log: {pattern!r} not found")
    return g[1]


def _gb(text: str) -> float:
    return float(text[:-1]) * {"K": 1e-6, "M": 1e-3, "G": 1.0, "T": 1e3}[text[-1]]


def solve(cfg: dict, work: Path, cores: int, tag: str) -> dict:
    path = work / f"config-{tag}.json"
    path.write_text(json.dumps(cfg, indent=1))
    t0 = time.time()
    lines = []
    with subprocess.Popen(["palace", "-np", str(cores), str(path)], cwd=work, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
        for line in proc.stdout:
            lines.append(line)
            if PROGRESS.search(line):
                print(f"[{tag}] {line.rstrip()}", flush=True)
    log = "".join(lines)
    (work / f"palace-{tag}.log").write_text(log)
    if proc.returncode:
        sys.stdout.write("".join(lines[-20:]))
        raise RuntimeError(f"palace {tag} exited with {proc.returncode}")
    peak = log[log.rindex("Peak Memory"):]
    return {"runtime_s": round(time.time() - t0, 1),
            "unknowns": int(_log_value(log, r"ND \(p = \d+\): (\d+)")),
            "memory_peak_gb": round(_gb(re.search(r"\nTotal\s.*\s(\S+)\s*\n", peak)[1]), 2),
            "version": _log_value(log, r"Git changeset ID: (\S+)")}


def _cpu() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def write(case: Case, out: Path, model: model3d.Model, m: dict, levels: list[dict],
          nets: list, cores: int) -> None:
    last = levels[-1]
    nets[-1].write_touchstone(str(out / "reference"), form="ri")
    entries = []
    for k, lv in enumerate(levels):
        e = {"label": f"element order {lv['order']}", "unknowns": lv["unknowns"]}
        if k:
            e["max_ds_to_previous"] = round(max_ds(nets[k], nets[k - 1], case.band)[0], 5)
        entries.append(e)
    unmodelled = [f"{mt.name}: {what}" for mt in model.materials.values()
                  for what, v in (("dispersion", mt.dispersion), ("roughness", mt.roughness)) if v]
    meta = {
        "solver": "palace",
        "version": last["version"],
        "date": _dt.date.today().isoformat(),
        "ports": case.reference_ports_key,
        "model": {
            "conductors": "volumes with bulk conductivity, meshed inside",
            "boundaries": {"open": f"absorbing, order {ABSORBING_ORDER}",
                           "conductor": "Conductivity boundary (half space)", "pec": "PEC"},
            "box_um": {k: [round(v, 3) for v in m["box"][k]] for k in "xyz"},
            "box_rule": {"D": "max(structure height, smaller lateral structure extent)",
                         "margin_D": BOX_MARGIN, "air_D": BOX_AIR},
            "mesh": {"tets": m["tets"], "generator": "gmsh HXT, single-threaded",
                     "rule": {"across": ACROSS, "growth": GROWTH, "per_wavelength": PER_WAVELENGTH,
                              "feature_tol": FEATURE_TOL},
                     "conductors": m["conductors"]},
            "ports": "lumped, R = z0, every port excited in turn",
            "sweep": {"adaptive_tol": ADAPTIVE_TOL, "adaptive_max_samples": ADAPTIVE_MAX_SAMPLES,
                      "points": FREQ_POINTS},
            "linear": LINEAR,
            "dielectrics": "er, loss tangent and bulk conductivity, constant over frequency",
            "not_modelled": unmodelled,
            "extra_order": f"order {EXTRA_ORDER} when order {ORDERS[-1]} vs {ORDERS[-2]} is above "
                           "the bound and its estimated memory fits",
            "level_runtime_s": [lv["runtime_s"] for lv in levels],
            "level_memory_gb": [lv["memory_peak_gb"] for lv in levels],
        },
        "levels": entries,
        "converged_bound": CONVERGED_BOUND,
        "runtime_s": last["runtime_s"],
        "memory_peak_gb": last["memory_peak_gb"],
        "machine": {"cpu": _cpu(), "cores": cores},
    }
    (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")


def _memory_limit_gb() -> float:
    """The job's memory limit: the cgroup limit where there is one, else the machine's memory."""
    try:
        return int(Path("/sys/fs/cgroup/memory.max").read_text()) / 1e9
    except (OSError, ValueError):
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9


def _memory_ratio(p: int, q: int) -> float:
    """Memory at order q over order p on one mesh: global Nedelec unknowns (about 1.2 edges and
    2 faces per tet) times the unknowns per element (dense element blocks). Megtron 1p0, p3 -> p4:
    estimated 9.0 GB, measured 8.6 GB."""
    def nd_per_tet(k): return 1.2 * k + 2 * k * (k - 1) + k * (k - 1) * (k - 2) / 2
    def per_element(k): return k * (k + 2) * (k + 3) / 2
    return nd_per_tet(q) * per_element(q) / (nd_per_tet(p) * per_element(p))


def _previous(out: Path, meta: dict | None, order: int, tets: int) -> dict | None:
    """A level already solved on the same mesh in out (meta.yaml read before this run, and
    p<order>/port-S.csv)."""
    try:
        k = [lv["label"] for lv in meta["levels"]].index(f"element order {order}")
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if meta["model"]["mesh"]["tets"] != tets or not (out / f"p{order}" / "port-S.csv").exists():
        return None
    return {"runtime_s": meta["model"]["level_runtime_s"][k],
            "memory_peak_gb": meta["model"].get("level_memory_gb", [meta["memory_peak_gb"]] * (k + 1))[k],
            "unknowns": meta["levels"][k]["unknowns"], "version": meta["version"]}


def reference(case: Case, cores: int, out: Path | None = None) -> None:
    import skrf
    out = Path(out) if out else case.directory / REFERENCE_DIR / "palace"
    out.mkdir(parents=True, exist_ok=True)
    model = model3d.build(case)
    work = Path(tempfile.mkdtemp(prefix=f"palace-{case.id}-"))
    t0 = time.time()
    mesh_path = work / "mesh.msh"
    m = mesh(model, mesh_path)
    print(f"[mesh] {m['tets']} tets in {time.time() - t0:.0f} s, conductors {m['conductors']}, "
          f"box {m['box']}", flush=True)
    levels, nets = [], []
    orders = list(ORDERS)
    meta = yaml.safe_load((out / "meta.yaml").read_text(encoding="utf-8")) if (out / "meta.yaml").exists() else None
    for order in orders:
        post = out / f"p{order}"
        lv = _previous(out, meta, order, m["tets"])
        if lv:
            print(f"[p{order}] taken from {post}", flush=True)
        else:
            lv = solve(config(model, m, mesh_path, order, post), work, cores, f"p{order}")
        f, s = read_s(post, len(model.ports))
        nets.append(skrf.Network(frequency=skrf.Frequency.from_f(f, unit="hz"), s=s,
                                 z0=[p.z0_ohm for p in model.ports]))
        levels.append({"order": order, **lv})
        write(case, out, model, m, levels, nets, cores)
        print(f"[p{order}] {lv['unknowns']} unknowns, {lv['runtime_s']:.0f} s, "
              f"{lv['memory_peak_gb']} GB, written to {out}", flush=True)
        if order == ORDERS[-1]:
            ds = max_ds(nets[-1], nets[-2], case.band)[0]
            gb = lv["memory_peak_gb"] * _memory_ratio(order, EXTRA_ORDER)
            fits = gb < _memory_limit_gb()
            print(f"[p{order}] max |dS| to p{ORDERS[-2]} {ds:.4f}; p{EXTRA_ORDER} estimated "
                  f"{gb:.1f} GB of {_memory_limit_gb():.1f} GB", flush=True)
            if ds > CONVERGED_BOUND and fits:
                orders.append(EXTRA_ORDER)
