"""Layer stacks as plain data.

One z axis for everything: z = 0 is the bottom of the lowest dielectric and all
heights are absolute, in um. Dielectrics are listed bottom to top and tile the
column without gaps; conductors (metals and vias) carry their own [bottom, top]
and are embedded in that column. A consumer never re-derives an offset.

A conformal shell coats the drawn shape of one conductor: grown laterally by
side_um (mitred corners, see SHELL_MITRE_LIMIT) from the bottom of the conductor
up to top_um above its top (shell_z). Inside it the shell replaces the planar
dielectrics; the conductor itself stays conductor. What covers the floor between
the traces is part of the planar column: its dielectrics end where the floor
coating ends.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STACK_DIR = ROOT / "stacks"

# Mitre limit of the lateral growth of a conformal shell, in multiples of side_um:
# the corner rule of gdspy.offset(join="miter", tolerance=2), with which the
# gds2palace stackups derive their shell layers (SIZE).
SHELL_MITRE_LIMIT = 2.0


@dataclass(frozen=True)
class Stack:
    path: Path
    raw: dict

    @property
    def name(self) -> str:
        return self.raw["name"]

    @property
    def dielectrics(self) -> list[dict]:
        return self.raw["dielectrics"]

    @property
    def conductors(self) -> list[dict]:
        return self.raw["conductors"]

    @property
    def conformal(self) -> list[dict]:
        return self.raw.get("conformal", [])

    @property
    def boundaries(self) -> dict:
        return self.raw["boundaries"]

    @property
    def height_um(self) -> float:
        return sum(float(d["thickness_um"]) for d in self.dielectrics)

    def dielectric_spans(self) -> list[tuple[dict, float, float]]:
        out, z = [], 0.0
        for d in self.dielectrics:
            out.append((d, z, z + float(d["thickness_um"])))
            z += float(d["thickness_um"])
        return out

    def conductor(self, name: str) -> dict:
        for c in self.conductors:
            if c["name"] == name:
                return c
        raise KeyError(f"{name!r} is not a conductor of stack {self.name!r}")

    def has(self, name: str) -> bool:
        return any(c["name"] == name for c in self.conductors)

    def by_gds(self) -> dict[tuple[int, int], str]:
        return {tuple(c["gds"]): c["name"] for c in self.conductors}

    def z(self, name: str) -> tuple[float, float]:
        lo, hi = self.conductor(name)["z_um"]
        return float(lo), float(hi)

    def shell_z(self, shell: dict) -> tuple[float, float]:
        """[bottom, top] of a conformal shell: from the bottom of the conductor it
        coats to top_um above its top."""
        lo, hi = self.z(shell["over"])
        return lo, hi + float(shell["top_um"])

    def check(self) -> list[str]:
        """Consistency the schema cannot express."""
        out = []
        top = self.height_um
        names = [c["name"] for c in self.conductors]
        if len(set(names)) != len(names):
            out.append("conductor names are not unique")
        gds = [tuple(c["gds"]) for c in self.conductors]
        if len(set(gds)) != len(gds):
            out.append("two conductors share a GDS layer")
        for c in self.conductors:
            lo, hi = c["z_um"]
            if not 0.0 <= lo < hi:
                out.append(f"{c['name']}: z_um must satisfy 0 <= bottom < top")
            if hi > top + 1e-9 and self.boundaries["top"] != "open":
                out.append(f"{c['name']}: top {hi} um is above the closed top boundary")
        for s in self.conformal:
            if not self.has(s["over"]):
                out.append(f"conformal {s['name']}: unknown conductor {s['over']!r}")
        for e in self.dielectrics + self.conductors + self.conformal:
            p = e["provenance"]
            for field in (p if isinstance(p, dict) else {}):
                if field != "default" and field not in e:
                    out.append(f"{e['name']}: provenance names {field!r}, which is not set")
        return out


def load_stack(ref: str, base: Path | None = None) -> Stack:
    """A stack is named (stacks/<name>.yaml) or given as a path relative to the
    case directory, for a stack that belongs to one board only."""
    path = (base / ref) if base is not None and ref.endswith(".yaml") else STACK_DIR / f"{ref}.yaml"
    return Stack(path, yaml.safe_load(path.read_text(encoding="utf-8")))
