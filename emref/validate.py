"""Everything CI checks before a case is accepted.

Errors block; warnings are printed and documented in the case but do not block
(measured data are noisy, so a passivity excursion of a few 1e-3 is a property of
the measurement, not a defect of the entry).
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from .case import Case, sha256, validate_schema
from .stack import Stack

PASSIVITY_TOL = 1e-2
RECIPROCITY_TOL = 2e-2
TOUCH_UM = 1e-3


def check_stack(stack: Stack) -> list[str]:
    errors = validate_schema(stack.raw, "stack")
    return errors or stack.check()


def _union(polys: list) -> object:
    return unary_union([Polygon(p).buffer(0) for p in polys if len(p) >= 3])


def check_ports(case: Case, layers: dict[str, list]) -> list[str]:
    """Every port must sit on the metal it claims to drive."""
    out = []
    stack = case.stack
    if len(case.ports()) != len({p.name for p in case.ports()}):
        out.append("port names are not unique")
    for p in case.ports():
        for name in (p.layer, p.reference):
            if name != "bottom_boundary" and not stack.has(name):
                out.append(f"port {p.name}: unknown conductor {name!r}")
        if p.layer not in layers:
            out.append(f"port {p.name}: nothing drawn on {p.layer}")
            continue
        seg = LineString(p.segment)
        metal = _union(layers[p.layer])
        if p.kind == "edge":
            if seg.distance(metal.boundary) > TOUCH_UM or seg.length - seg.intersection(
                    metal.boundary.buffer(TOUCH_UM)).length > TOUCH_UM:
                out.append(f"port {p.name}: segment does not lie on an edge of {p.layer}")
        else:
            if seg.distance(metal) > TOUCH_UM:
                out.append(f"port {p.name}: segment does not touch {p.layer}")
            if p.reference != "bottom_boundary":
                if p.reference not in layers:
                    out.append(f"port {p.name}: nothing drawn on {p.reference}")
                elif seg.distance(_union(layers[p.reference])) > TOUCH_UM:
                    out.append(f"port {p.name}: segment is not above {p.reference}")
                elif stack.z(p.reference)[1] > stack.z(p.layer)[0]:
                    out.append(f"port {p.name}: {p.reference} is not below {p.layer}")
    return out


def check_measurement(case: Case) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    f_min, f_max = case.band
    for m in case.measurements():
        if not m.path.exists():
            errors.append(f"measurement {m.label}: {m.path.name} missing")
            continue
        net = m.network()
        tag = f"measurement {m.label}"
        if net.nports != len(case.ports()):
            errors.append(f"{tag}: {net.nports} ports, case defines {len(case.ports())}")
        if not np.all(np.diff(net.f) > 0):
            errors.append(f"{tag}: frequency axis is not strictly increasing")
        if net.f[0] > f_min * 1.001 or net.f[-1] < f_max * 0.999:
            errors.append(f"{tag}: data do not cover the band")
        if not np.all(np.isfinite(net.s)):
            errors.append(f"{tag}: non-finite entries")
            continue
        sv = np.linalg.svd(net.s, compute_uv=False).max(axis=1)
        if sv.max() > 1 + PASSIVITY_TOL:
            warnings.append(f"{tag}: max singular value {sv.max():.4f} at "
                            f"{net.f[sv.argmax()] / 1e9:.2f} GHz")
        asym = np.abs(net.s - np.swapaxes(net.s, 1, 2)).max()
        if asym > RECIPROCITY_TOL:
            warnings.append(f"{tag}: max |S - S^T| = {asym:.3f}")
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
    errors += check_ports(case, layers)
    errors += check_sources(case)
    e, warnings = check_measurement(case)
    return errors + e, warnings
