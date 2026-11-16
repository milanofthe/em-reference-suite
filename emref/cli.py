"""python -m emref {validate,build,check,list,model,reference}

build  writes every generated file (figures, case READMEs, the gallery).
check  validates and rebuilds in memory; fails when a committed file is stale.
model  prints the solver-neutral 3D model of a case (emref.model3d).
reference  computes a case's 3D reference with a solver driver (emref.solvers).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from . import render, report
import yaml

from .case import Case, iter_cases, load_case, validate_schema
from .stack import ROOT, STACK_DIR, load_stack
from .validate import check_case, check_stack


def _select(ids: list[str]) -> list[Case]:
    return [load_case(i) for i in ids] if ids else iter_cases()


def _artefacts(case: Case, out: Path) -> dict[Path, bytes]:
    """Generated files of one case, keyed by their committed path."""
    render.layout(case, out / report.LAYOUT_SVG)
    render.sparams(case, out / report.SPARAMS_SVG)
    # matplotlib writes platform line endings; the committed files are LF everywhere
    files = {case.directory / n: (out / n).read_bytes().replace(b"\r\n", b"\n")
             for n in (report.LAYOUT_SVG, report.SPARAMS_SVG)}
    files[case.directory / "README.md"] = report.case_readme(case).encode()
    return files


def _generated(cases: list[Case]) -> dict[Path, bytes]:
    files: dict[Path, bytes] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for c in cases:
            out = Path(tmp) / c.id
            out.mkdir()
            files.update(_artefacts(c, out))
    files[ROOT / "README.md"] = report.top_readme(iter_cases(), load_candidates()).encode()
    return files


def load_candidates() -> list[dict]:
    return yaml.safe_load((ROOT / "candidates.yaml").read_text(encoding="utf-8"))


def cmd_validate(args) -> int:
    failed = False
    for e in validate_schema(load_candidates(), "candidates"):
        print(f"candidates.yaml: {e}")
        failed = True
    for path in sorted(STACK_DIR.glob("*.yaml")):
        for e in check_stack(load_stack(path.stem)):
            print(f"stacks/{path.name}: {e}")
            failed = True
    for c in _select(args.ids):
        errors, warnings = check_case(c)
        for w in warnings:
            print(f"{c.id}: warning: {w}")
        for e in errors:
            print(f"{c.id}: error: {e}")
        failed |= bool(errors)
        if not errors:
            print(f"{c.id}: ok, {report.status_line(c).replace('**', '')}")
    return int(failed)


def cmd_build(args) -> int:
    for path, data in _generated(_select(args.ids)).items():
        if not path.exists() or path.read_bytes() != data:
            path.write_bytes(data)
            print(f"wrote {path.relative_to(ROOT)}")
    return 0


def cmd_check(args) -> int:
    rc = cmd_validate(args)
    stale = [p for p, data in _generated(_select(args.ids)).items()
             if not p.exists() or p.read_bytes() != data]
    for p in stale:
        print(f"stale: {p.relative_to(ROOT)} (run python -m emref build)")
    return rc or int(bool(stale))


def cmd_list(args) -> int:
    for c in iter_cases():
        print(f"{c.id:40s} {report.summary_line(c)}")
    return 0


def _boundary(b) -> str:
    return f"conductor {b.sigma:g} S/m" if b.kind == "conductor" else b.kind


def cmd_model(args) -> int:
    from . import model3d
    try:
        m = model3d.build(load_case(args.id))
    except ValueError as e:
        print(e)
        return 1
    (x0, y0, z0), (x1, y1, z1) = m.bbox
    print(f"{m.case}: band {m.band_hz[0]:g}-{m.band_hz[1]:g} Hz, bbox x {x0:g}..{x1:g}, "
          f"y {y0:g}..{y1:g}, z {z0:g}..{z1:g} um")
    print(f"boundaries: bottom {_boundary(m.boundaries['bottom'])} at z {m.boundaries['bottom'].z:g}, "
          f"top {_boundary(m.boundaries['top'])}")
    print()
    print("solids (higher priority wins; xy 'box': spans the simulation box)")
    print(f"{'name':24s} {'role':10s} {'prio':>4s} {'z [um]':>19s} {'n':>4s} {'vertices':>8s}  xy [um]")
    groups: dict[tuple, list] = {}
    for s in m.solids:
        groups.setdefault((s.name, s.role, s.priority, s.z0, s.z1), []).append(s)
    for (name, role, prio, lo, hi), ss in groups.items():
        pts = [xy for s in ss if s.outer for xy in s.outer]
        nv = sum(len(s.outer) + sum(len(h) for h in s.holes) for s in ss if s.outer)
        xy = "box" if not pts else (f"{min(x for x, _ in pts):g}..{max(x for x, _ in pts):g}, "
                                    f"{min(y for _, y in pts):g}..{max(y for _, y in pts):g}")
        print(f"{name:24s} {role:10s} {prio:4d} {lo:9.4f}-{hi:9.4f} {len(ss):4d} {nv:8d}  {xy}")
    print()
    print("materials")
    print(f"{'name':24s} {'kind':10s} {'er':>6s} {'tand':>7s} {'sigma [S/m]':>11s}  dispersion")
    for mt in m.materials.values():
        sigma = "PEC" if mt.pec else f"{mt.sigma:g}"
        disp = "-" if mt.dispersion is None else "DS f_ref {:g}, {:g}..{:g} Hz".format(*mt.dispersion)
        er, tand = ("-", "-") if mt.kind == "conductor" else (f"{mt.er:g}", f"{mt.tand:g}")
        print(f"{mt.name:24s} {mt.kind:10s} {er:>6s} {tand:>7s} {sigma:>11s}  {disp}")
    print()
    print("ports (lumped, vertical rectangle)")
    print(f"{'name':5s} {'minus -> plus':28s} {'z0':>5s}  corners [um]")
    for p in m.ports:
        corners = " ".join("({:g}, {:g}, {:g})".format(*c) for c in p.corners)
        print(f"{p.name:5s} {p.minus + ' -> ' + p.plus:28s} {p.z0_ohm:5g}  {corners}")
    return 0


def cmd_reference(args) -> int:
    from .solvers import palace
    palace.reference(load_case(args.id), args.cores, args.out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="emref")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("validate", cmd_validate), ("build", cmd_build), ("check", cmd_check)):
        sp = sub.add_parser(name)
        sp.add_argument("ids", nargs="*")
        sp.set_defaults(fn=fn)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    sp = sub.add_parser("model")
    sp.add_argument("id")
    sp.set_defaults(fn=cmd_model)
    sp = sub.add_parser("reference")
    sp.add_argument("id")
    sp.add_argument("solver", choices=["palace"])
    sp.add_argument("--cores", type=int, required=True)
    sp.add_argument("--out", type=Path)
    sp.set_defaults(fn=cmd_reference)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
