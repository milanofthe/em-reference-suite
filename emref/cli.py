"""python -m emref {validate,build,check,list}

build  writes every generated file (figures, case READMEs, the gallery).
check  validates and rebuilds in memory; fails when a committed file is stale.
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="emref")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("validate", cmd_validate), ("build", cmd_build), ("check", cmd_check)):
        sp = sub.add_parser(name)
        sp.add_argument("ids", nargs="*")
        sp.set_defaults(fn=fn)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
