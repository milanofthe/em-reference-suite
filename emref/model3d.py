"""The solver-neutral 3D model of a case: what every driver builds, nothing it solves.

Units um, frequencies Hz. Space not covered by a solid is vacuum. Solids overlap
and the higher priority wins (dielectric < shell < conductor = via); the drivers
apply it (gmsh fragments, CSXCAD priorities). A solid without an outline spans the
simulation box laterally. The simulation box itself (air above and below, lateral
margin) is not part of the model: it is a parameter of each driver's convergence
series.
"""
from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry.polygon import orient

from .case import Case, union
from .stack import SHELL_MITRE_LIMIT

PRIORITY = {"dielectric": 0, "shell": 1, "conductor": 2, "via": 2}

XY = tuple[float, float]
XYZ = tuple[float, float, float]


@dataclass(frozen=True)
class Solid:
    """An extrusion of a polygon (outer ring counter-clockwise, holes clockwise,
    rings not closed) from z0 to z1."""
    name: str                       # the stack entry it comes from
    material: str
    role: str                       # dielectric | shell | conductor | via
    z0: float
    z1: float
    outer: tuple[XY, ...] | None    # None: spans the simulation box laterally
    holes: tuple[tuple[XY, ...], ...] = ()

    @property
    def priority(self) -> int:
        return PRIORITY[self.role]


@dataclass(frozen=True)
class Material:
    name: str
    kind: str                       # dielectric | conductor
    er: float = 1.0
    tand: float = 0.0
    sigma: float = 0.0              # S/m; a dielectric's bulk conductivity (Si substrate)
    pec: bool = False
    # Djordjevic-Sarkar (f_ref, f_low, f_high) in Hz, er and tand stated at f_ref.
    # None: er and tand at all frequencies.
    dispersion: tuple[float, float, float] | None = None
    roughness: dict | None = None   # as in the stack: model, rq_um, sides


@dataclass(frozen=True)
class Boundary:
    kind: str                       # pec | conductor | open
    z: float                        # plane of the boundary
    sigma: float | None = None      # conductor: a solid good conductor below / above


@dataclass(frozen=True)
class LumpedPort:
    """A vertical rectangle on the port segment between two conductors."""
    name: str
    corners: tuple[XYZ, XYZ, XYZ, XYZ]   # a-bottom, b-bottom, b-top, a-top
    direction: XYZ                  # unit vector from the minus to the plus conductor
    minus: str                      # conductor name or bottom_boundary
    plus: str
    z0_ohm: float


@dataclass(frozen=True)
class Model:
    case: str
    solids: tuple[Solid, ...]
    materials: dict[str, Material]
    boundaries: dict[str, Boundary]  # bottom, top
    ports: tuple[LumpedPort, ...]
    bbox: tuple[XYZ, XYZ]           # of the structure: laterally bounded solids and ports
    band_hz: tuple[float, float]


def _rings(region) -> list[tuple[tuple[XY, ...], tuple[tuple[XY, ...], ...]]]:
    parts = getattr(region, "geoms", [region])
    out = []
    for poly in parts:
        if poly.is_empty:
            continue
        poly = orient(poly, 1.0)
        out.append((tuple(poly.exterior.coords[:-1]),
                    tuple(tuple(r.coords[:-1]) for r in poly.interiors)))
    return out


def _material(entry: dict, kind: str) -> Material:
    dispersion = None
    if entry.get("model", "constant") == "djordjevic_sarkar":
        dispersion = (float(entry["f_ref_hz"]), float(entry["f_low_hz"]), float(entry["f_high_hz"]))
    if kind == "conductor":
        return Material(entry["name"], kind, sigma=float(entry.get("sigma_s_per_m", 0.0)),
                        pec=bool(entry.get("pec", False)), roughness=entry.get("roughness"))
    return Material(entry["name"], kind, er=float(entry["er"]), tand=float(entry["tand"]),
                    sigma=float(entry.get("sigma_s_per_m", 0.0)), dispersion=dispersion)


def _boundary(b, z: float) -> Boundary:
    if isinstance(b, str):
        return Boundary(b, z)
    return Boundary("conductor", z, float(b["sigma_s_per_m"]))


def _ports(case: Case) -> tuple[LumpedPort, ...]:
    stack = case.stack
    out = []
    for p in case.reference_ports():
        if p.kind != "vertical":
            raise ValueError(f"{case.id}: port {p.name} is of kind {p.kind}; the 3D references "
                             "need lumped ports (reference_ports, kind vertical)")
        lo = 0.0 if p.reference == "bottom_boundary" else stack.z(p.reference)[1]
        hi = stack.z(p.layer)[0]
        (ax, ay), (bx, by) = p.segment
        out.append(LumpedPort(p.name, ((ax, ay, lo), (bx, by, lo), (bx, by, hi), (ax, ay, hi)),
                              (0.0, 0.0, 1.0), p.reference, p.layer, float(p.z0_ohm)))
    return tuple(out)


def build(case: Case) -> Model:
    stack = case.stack
    layers, _ = case.geometry()
    solids: list[Solid] = []
    materials: list[Material] = []

    outline = case.outline() if stack.raw.get("lateral", "infinite") == "outline" else None
    footprint = _rings(union(outline)) if outline else [(None, ())]
    for d, lo, hi in stack.dielectric_spans():
        materials.append(_material(d, "dielectric"))
        solids += [Solid(d["name"], d["name"], "dielectric", lo, hi, outer, holes)
                   for outer, holes in footprint]

    for s in stack.conformal:
        if s["over"] not in layers:
            continue
        materials.append(_material(s, "dielectric"))
        region = union(layers[s["over"]]).buffer(float(s["side_um"]), join_style="mitre",
                                                  mitre_limit=SHELL_MITRE_LIMIT)
        lo, hi = stack.shell_z(s)
        solids += [Solid(s["name"], s["name"], "shell", lo, hi, outer, holes)
                   for outer, holes in _rings(region)]

    for c in stack.conductors:
        if c["name"] not in layers:
            continue
        materials.append(_material(c, "conductor"))
        region = union(layers[c["name"]])
        if "plating_um" in c:
            region = region.difference(region.buffer(-float(c["plating_um"]), join_style="mitre"))
        lo, hi = stack.z(c["name"])
        role = "via" if c["kind"] == "via" else "conductor"
        solids += [Solid(c["name"], c["name"], role, lo, hi, outer, holes)
                   for outer, holes in _rings(region)]

    names = [m.name for m in materials]
    if len(set(names)) != len(names):
        raise ValueError(f"{case.id}: material names are not unique: {names}")

    ports = _ports(case)
    pts = [xy for s in solids if s.outer for xy in s.outer]
    pts += [c[:2] for p in ports for c in p.corners]
    zs = [z for s in solids for z in (s.z0, s.z1)]
    bbox = ((min(x for x, _ in pts), min(y for _, y in pts), min(zs)),
            (max(x for x, _ in pts), max(y for _, y in pts), max(zs)))
    boundaries = {"bottom": _boundary(stack.boundaries["bottom"], 0.0),
                  "top": _boundary(stack.boundaries["top"], stack.height_um)}
    return Model(case.id, tuple(solids), {m.name: m for m in materials}, boundaries, ports,
                 bbox, case.band)
