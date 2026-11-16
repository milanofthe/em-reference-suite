"""Everything CI checks before a case is accepted.

Errors block; warnings are printed and documented in the case but do not block
(measured data are noisy, so a passivity excursion of a few 1e-3 is a property of
the measurement, not a defect of the entry).
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import LineString

from .case import Case, sha256, union, validate_schema
from .stack import Stack

PASSIVITY_TOL = 1e-2
RECIPROCITY_TOL = 2e-2
TOUCH_UM = 1e-3


def check_stack(stack: Stack) -> list[str]:
    errors = validate_schema(stack.raw, "stack")
    return errors or stack.check()


def check_ports(case: Case, ports: list, layers: dict[str, list]) -> list[str]:
    """Every port must sit on the metal it claims to drive."""
    out = []
    stack = case.stack
    if len(ports) != len({p.name for p in ports}):
        out.append("port names are not unique")
    for p in ports:
        for name in (p.layer, p.reference):
            if name not in ("bottom_boundary", "common_node") and not stack.has(name):
                out.append(f"port {p.name}: unknown conductor {name!r}")
        if p.layer not in layers:
            out.append(f"port {p.name}: nothing drawn on {p.layer}")
            continue
        seg = LineString(p.segment)
        metal = union(layers[p.layer])
        if p.kind in ("edge", "line"):
            if seg.distance(metal.boundary) > TOUCH_UM or seg.length - seg.intersection(
                    metal.boundary.buffer(TOUCH_UM)).length > TOUCH_UM:
                out.append(f"port {p.name}: segment does not lie on an edge of {p.layer}")
        else:
            if seg.distance(metal) > TOUCH_UM:
                out.append(f"port {p.name}: segment does not touch {p.layer}")
            if p.reference not in ("bottom_boundary", "common_node"):
                if p.reference not in layers:
                    out.append(f"port {p.name}: nothing drawn on {p.reference}")
                elif seg.distance(union(layers[p.reference])) > TOUCH_UM:
                    out.append(f"port {p.name}: segment is not above {p.reference}")
                elif stack.z(p.reference)[1] > stack.z(p.layer)[0]:
                    out.append(f"port {p.name}: {p.reference} is not below {p.layer}")
            elif p.reference == "bottom_boundary" and stack.boundaries["bottom"] == "open":
                out.append(f"port {p.name}: starts on an open bottom boundary")
    return out


def check_network(tag: str, net, nports: int, band: tuple[float, float]
                  ) -> tuple[list[str], list[str]]:
    """A Touchstone file against the case: port count, band, finiteness; passivity
    and reciprocity as warnings."""
    errors, warnings = [], []
    f_min, f_max = band
    if net.nports != nports:
        errors.append(f"{tag}: {net.nports} ports, case defines {nports}")
    if not np.all(np.diff(net.f) > 0):
        errors.append(f"{tag}: frequency axis is not strictly increasing")
    if net.f[0] > f_min * 1.001 or net.f[-1] < f_max * 0.999:
        errors.append(f"{tag}: data do not cover the band")
    if not np.all(np.isfinite(net.s)):
        errors.append(f"{tag}: non-finite entries")
        return errors, warnings
    sv = np.linalg.svd(net.s, compute_uv=False).max(axis=1)
    if sv.max() > 1 + PASSIVITY_TOL:
        warnings.append(f"{tag}: max singular value {sv.max():.4f} at "
                        f"{net.f[sv.argmax()] / 1e9:.2f} GHz")
    asym = np.abs(net.s - np.swapaxes(net.s, 1, 2)).max()
    if asym > RECIPROCITY_TOL:
        warnings.append(f"{tag}: max |S - S^T| = {asym:.3f}")
    return errors, warnings


def check_measurement(case: Case) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    for m in case.measurements():
        if not m.path.exists():
            errors.append(f"measurement {m.label}: {m.path.name} missing")
            continue
        e, w = check_network(f"measurement {m.label}", m.network(), len(case.ports()), case.band)
        errors += e
        warnings += w
    return errors, warnings


def _check_series(tag: str, series: list[dict]) -> list[str]:
    out = []
    if "max_ds_to_previous" in series[0]:
        out.append(f"{tag}: the first level has no previous level")
    if any("max_ds_to_previous" not in lv for lv in series[1:]):
        out.append(f"{tag}: max_ds_to_previous missing after the first level")
    return out


def converged(meta: dict) -> bool:
    """max |dS| of the last level of each series within converged_bound."""
    return all(s[-1]["max_ds_to_previous"] <= meta["converged_bound"]
               for s in (meta["levels"], meta.get("box_series", [{"max_ds_to_previous": 0.0}])))


def check_references(case: Case) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    ports = case.reference_ports()
    if case.references() and any(p.kind != "vertical" for p in ports):
        errors.append(f"{case.reference_ports_key}: the references need lumped ports "
                      "(kind vertical); add reference_ports")
        return errors, warnings
    for r in case.references():
        tag = f"reference {r.solver}"
        e = validate_schema(r.meta, "reference")
        if e:
            errors += [f"{tag}: meta.yaml: {x}" for x in e]
            continue
        if r.meta["solver"] != r.solver:
            errors.append(f"{tag}: meta.yaml names solver {r.meta['solver']!r}")
        if r.meta["ports"] != case.reference_ports_key:
            errors.append(f"{tag}: uses {r.meta['ports']}, the case prescribes "
                          f"{case.reference_ports_key}")
        series = [("levels", r.meta["levels"])]
        series += [("box_series", r.meta["box_series"])] if "box_series" in r.meta else []
        e = [x for name, s in series for x in _check_series(f"{tag}: {name}", s)]
        errors += e
        if not e and not converged(r.meta):
            warnings.append(f"{tag}: not converged, last max |dS| above "
                            f"{r.meta['converged_bound']:g}")
        files = r.touchstone()
        if [f.name for f in files] != [f"reference.s{len(ports)}p"]:
            errors.append(f"{tag}: expected exactly reference.s{len(ports)}p, found "
                          f"{[f.name for f in files]}")
            continue
        net = r.network()
        e, w = check_network(tag, net, len(ports), case.band)
        errors += e
        warnings += w
        z0 = np.array([p.z0_ohm for p in ports])
        if not e and not np.allclose(net.z0, z0[None, :]):
            errors.append(f"{tag}: reference impedances differ from the port Z0 {z0.tolist()}")
    return errors, warnings


def check_sources(case: Case) -> list[str]:
    """The committed copies still hash to what the upstream files hashed to."""
    out = []
    hashes = {f["path"]: f["sha256"] for f in case.raw["source"].get("files", [])}
    pairs = [(m.meta.get("source_file"), m.path) for m in case.measurements()]
    pairs.append((case.raw["layout"].get("source_file"), case.gds_path))
    for upstream, local in pairs:
        if upstream is None or not local.exists():
            continue
        if upstream not in hashes:
            out.append(f"{local.name}: upstream {upstream!r} not listed under source.files")
        elif sha256(local) != hashes[upstream]:
            out.append(f"{local.name}: differs from upstream {upstream}")
    return out


def check_case(case: Case) -> tuple[list[str], list[str]]:
    errors = validate_schema(case.raw, "case")
    if errors:
        return errors, []
    if case.id != case.directory.name:
        errors.append(f"id {case.id!r} differs from directory name")
    errors += [f"stack: {e}" for e in check_stack(case.stack)]
    if errors:
        return errors, []
    layers, unknown = case.geometry()
    if unknown:
        errors.append(f"layout draws GDS layers the stack does not model: {unknown} "
                      f"(add them to the stack or to layout.ignore_layers)")
    errors += check_ports(case, case.ports(), layers)
    if "reference_ports" in case.raw:
        errors += [f"reference_ports: {e}" for e in check_ports(case, case.reference_ports(), layers)]
    if case.stack.raw.get("lateral") == "outline" and not case.outline():
        errors.append("stack has lateral: outline but the layout draws no outline "
                      "(layout.outline_layer)")
    errors += check_sources(case)
    if not case.measurements() and not case.references():
        errors.append("neither a measurement nor a reference")
    e, warnings = check_measurement(case)
    e2, w2 = check_references(case)
    return errors + e + e2, warnings + w2
