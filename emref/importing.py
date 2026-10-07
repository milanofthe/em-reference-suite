"""Shared helpers for the importers in importers/.

An importer turns one upstream source into one or more cases: it fetches the raw
data at a pinned commit, checks every file against its recorded SHA-256, and
writes layout.gds, the measured Touchstone and case.yaml. Running it again must
reproduce the committed files.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import urllib.request
from pathlib import Path

import gdstk
import yaml

from .stack import ROOT

CACHE = ROOT / ".cache"
GDS_EPOCH = _dt.datetime(1970, 1, 1)    # pinned, so a layout hashes to itself


def fetch(url: str, sha256: str) -> Path:
    """Download once into .cache/, then verify. A hash mismatch is fatal: the
    upstream file is not the one the case was built from."""
    path = CACHE / sha256[:16] / url.rsplit("/", 1)[-1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as r:
            path.write_bytes(r.read())
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != sha256:
        raise RuntimeError(f"{url}: sha256 {got} != {sha256}")
    return path


def write_gds(path: Path, cell_name: str, polygons: dict[tuple[int, int], list]) -> None:
    """Polygons keyed by (layer, datatype), coordinates in um."""
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell(cell_name)
    for (layer, datatype), polys in sorted(polygons.items()):
        for pts in polys:
            cell.add(gdstk.Polygon(pts, layer=layer, datatype=datatype))
    path.parent.mkdir(parents=True, exist_ok=True)
    lib.write_gds(str(path), timestamp=GDS_EPOCH)


def write_case(directory: Path, case: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(case, sort_keys=False, width=100, allow_unicode=False)
    (directory / "case.yaml").write_text(text, encoding="utf-8", newline="\n")


def write_touchstone(path: Path, net, comments: list[str]) -> None:
    """RI format, Hz, fixed precision, LF line endings."""
    lines = [f"! {c}" for c in comments]
    lines.append(f"# Hz S RI R {float(net.z0[0, 0].real):g}")
    n = net.nports
    for k, f in enumerate(net.f):
        # Touchstone 2-port order is S11 S21 S12 S22; generic N-port is row major.
        s = net.s[k].T.ravel() if n == 2 else net.s[k].ravel()
        lines.append(f"{f:.6f} " + " ".join(f"{v.real:+.9e} {v.imag:+.9e}" for v in s))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def gerber_copper(path: Path):
    """Copper of one Gerber layer as a shapely (Multi)Polygon in mm. Arcs are
    flattened by gerbonara; dark features are united, clear features are not
    supported (none of the imported boards use them)."""
    import warnings

    import shapely.geometry as sg
    import shapely.ops as so
    from gerbonara import GerberFile
    from gerbonara.utils import MM

    polys = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # Altium omits D01 on coordinate lines
        for obj in GerberFile.open(path).objects:
            for prim in obj.to_primitives(unit=MM):
                pts = list(prim.to_arc_poly().outline)
                if len(pts) > 2:
                    polys.append(sg.Polygon(pts).buffer(0))
    return so.unary_union(polys)
