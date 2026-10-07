"""A case: one measured structure, everything needed to simulate it, and the data.

``layout.gds`` is normative. Where the upstream source is a GDS it is committed
byte for byte, so its SHA-256 matches the one recorded under ``source.files``;
layers the stack does not model must be listed in ``layout.ignore_layers``, so
nothing in the file is dropped silently.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import gdstk
import yaml

from .stack import ROOT, Stack, load_stack

CASE_DIR = ROOT / "cases"
SCHEMA_DIR = ROOT / "schema"

# Source kinds that make a value something other than stated by the source.
WEAK = ("reconstructed", "fitted", "assumed")
# The values of a stack entry that a solver consumes, and so carry a provenance.
VALUE_FIELDS = ("thickness_um", "er", "tand", "sigma_s_per_m", "z_um", "pec",
                "roughness", "plating_um", "top_um", "side_um")


def _provenance(entry: dict) -> dict[str, str]:
    """Per value field, where it comes from."""
    p = entry["provenance"]
    fields = [f for f in VALUE_FIELDS if f in entry]
    if isinstance(p, str):
        return {f: p for f in fields}
    return {f: p.get(f, p["default"]) for f in fields}


@dataclass(frozen=True)
class Port:
    name: str
    kind: str                       # "edge" | "vertical"
    segment: tuple[tuple[float, float], tuple[float, float]]
    layer: str                      # the conductor the port drives
    reference: str                  # what it returns to: a conductor or "bottom_boundary"
    z0_ohm: float

    @property
    def midpoint(self) -> tuple[float, float]:
        (ax, ay), (bx, by) = self.segment
        return (ax + bx) / 2, (ay + by) / 2


@dataclass(frozen=True)
class Measurement:
    label: str
    path: Path
    meta: dict

    def network(self):
        import skrf
        return skrf.Network(str(self.path))


@dataclass(frozen=True)
class Case:
    directory: Path
    raw: dict

    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def title(self) -> str:
        return self.raw["title"]

    @property
    def open_questions(self) -> list[str]:
        return self.raw.get("open_questions", [])

    def status(self) -> tuple[str, list[str]]:
        """complete | reconstructed | incomplete, with the reasons. Derived from the
        data, never stated, so the label cannot drift from the content."""
        if self.open_questions:
            return "incomplete", list(self.open_questions)
        reasons = []
        if self.raw["layout"]["provenance"] == "reconstructed":
            reasons.append("layout reconstructed from dimensions or drawings")
        stack = self.stack
        used = set(self.geometry()[0])
        entries = [("dielectric", d) for d in stack.dielectrics]
        entries += [("conductor", c) for c in stack.conductors if c["name"] in used]
        entries += [("conformal", s) for s in stack.conformal if s["over"] in used]
        for kind, entry in entries:
            for field, how in _provenance(entry).items():
                if how in WEAK:
                    reasons.append(f"{kind} {entry['name']}: {field} {how}")
        return ("reconstructed" if reasons else "complete"), reasons

    @property
    def stack(self) -> Stack:
        return load_stack(self.raw["stack"], self.directory)

    @property
    def gds_path(self) -> Path:
        return self.directory / self.raw["layout"]["file"]

    @property
    def band(self) -> tuple[float, float]:
        b = self.raw["band"]
        return float(b["f_min_hz"]), float(b["f_max_hz"])

    def ports(self) -> list[Port]:
        out = []
        for p in self.raw["ports"]:
            seg = tuple(tuple(map(float, xy)) for xy in p["segment"])
            if p["kind"] == "vertical":
                out.append(Port(p["name"], "vertical", seg, p["to"], p["from"],
                                float(p.get("z0_ohm", 50.0))))
            else:
                out.append(Port(p["name"], "edge", seg, p["layer"], p["reference"],
                                float(p.get("z0_ohm", 50.0))))
        return out

    def measurements(self) -> list[Measurement]:
        return [Measurement(m["label"], self.directory / m["file"], m)
                for m in self.raw["measurements"]]

    def results(self) -> list[tuple[str, dict, Path]]:
        """(directory name, meta, touchstone) for every contributed solver result."""
        out = []
        for meta_path in sorted((self.directory / "results").glob("*/meta.yaml")):
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta.get("date"), (_dt.date, _dt.datetime)):
                meta["date"] = meta["date"].isoformat()[:10]
            out.append((meta_path.parent.name, meta, meta_path.parent / meta["touchstone"]))
        return out

    def cell(self) -> gdstk.Cell:
        lib = gdstk.read_gds(str(self.gds_path))
        name = self.raw["layout"].get("cell")
        if name is None:
            tops = lib.top_level()
            if len(tops) != 1:
                raise ValueError(f"{self.id}: {len(tops)} top cells, set layout.cell")
            return tops[0]
        return next(c for c in lib.cells if c.name == name)

    def geometry(self) -> tuple[dict[str, list], list[tuple[int, int]]]:
        """Polygons keyed by conductor name, flattened, plus every GDS layer that is
        drawn but neither modelled by the stack nor declared as ignored."""
        cell = self.cell().copy("flat").flatten()
        known = self.stack.by_gds()
        ignored = {tuple(x) for x in self.raw["layout"].get("ignore_layers", [])}
        layers: dict[str, list] = {}
        unknown = set()
        for poly in cell.polygons:
            key = (poly.layer, poly.datatype)
            if key in known:
                layers.setdefault(known[key], []).append(poly.points.tolist())
            elif key not in ignored:
                unknown.add(key)
        return layers, sorted(unknown)


def load_case(ref: str | Path) -> Case:
    directory = Path(ref)
    if not (directory / "case.yaml").exists():
        directory = CASE_DIR / str(ref)
    return Case(directory, yaml.safe_load((directory / "case.yaml").read_text(encoding="utf-8")))


def iter_cases() -> list[Case]:
    return [load_case(d) for d in sorted(CASE_DIR.glob("*/")) if (d / "case.yaml").exists()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_schema(instance: dict, name: str) -> list[str]:
    import jsonschema
    from referencing import Registry, Resource
    schemas = {p.name: json.loads(p.read_text(encoding="utf-8"))
               for p in SCHEMA_DIR.glob("*.schema.json")}
    registry = Registry().with_resources(
        (k, Resource.from_contents(v)) for k, v in schemas.items())
    validator = jsonschema.Draft202012Validator(schemas[f"{name}.schema.json"], registry=registry)
    return [f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
            for e in validator.iter_errors(instance)]
