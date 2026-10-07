"""IHP Open PDK, measured openEMS test cases on SG13G2 (removed from the PDK in 2025).

Source: https://github.com/IHP-GmbH/IHP-Open-PDK (Apache-2.0), libs.tech/openems/
testcase at commit 9439837, the last one before "removed old openEMS files". Two
measured structures:

- SG13_Octagon_L2n0: 2 nH octagonal inductor, two dies of wafer PQD701 W03 (the
  same wafer and, for x2y7, the same die as the L6n2 case), GSGSG pads and feeds
  removed by THRU de-embedding. The simplified GDS of the test case is the DUT.
- SG13_line: 880 um microstrip line on TopMetal2 over a Metal1 ground strip, pads
  de-embedded. No GDS was published; the geometry is that of the IHP model.

    python importers/ihp_openpdk.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emref.importing import github, write_case, write_gds  # noqa: E402
from emref.stack import ROOT  # noqa: E402

REPO = "IHP-GmbH/IHP-Open-PDK"
COMMIT = "943983786b04351feb20e33e8d3d76c0e0d87cce"
BASE = "ihp-sg13g2/libs.tech/openems/testcase"
L2N0_GDS = f"{BASE}/SG13_Octagon_L2n0/gds/L_2n0_simplified.gds"
L2N0_MEAS = {
    "die_x2y7": f"{BASE}/SG13_Octagon_L2n0/measured/meas_L3_2n0_THRU_deemb_GSGSG_PQD701W03Cx2y7.S2P",
    "die_x5y8": f"{BASE}/SG13_Octagon_L2n0/measured/meas_L3_2n0_THRU_deemb_GSGSG_PQD701W03Cx5y8.S2P",
}
LINE_MEAS = f"{BASE}/SG13_line/measured/LINE_880_corrected.S2P"
FILES = {
    L2N0_GDS: "4a8adee639a25b27023f093a1941e244ef93217a068219947b923cb32db72016",
    L2N0_MEAS["die_x2y7"]: "4a1febb9a4f2d0054c63ee800e57b2114bce8e44db642cf5b6ccb5f6c0739721",
    L2N0_MEAS["die_x5y8"]: "34ba71b0986d30a0941b3b50f38a7e8b2dec363ff1ba9f4e777dfc02e232f944",
    LINE_MEAS: "592ff1c9a872b055dfe9813eeeea6e8d959e1d49042b74929833d59ba50f6545",
}
SOURCE = {
    "title": "IHP Open PDK, openEMS test cases with measurements",
    "authors": ["IHP"],
    "url": f"https://github.com/{REPO}/tree/{COMMIT}/{BASE}",
    "commit": COMMIT,
    "license": "Apache-2.0",
}
# GDS layers of the stack, see stacks/ihp_sg13g2.yaml
METAL1, TOPMETAL2 = (8, 0), (134, 0)


def upstream(path: str) -> Path:
    return github(REPO, COMMIT, path, FILES[path])


def source(*paths: str) -> dict:
    return {**SOURCE, "files": [{"path": p, "sha256": FILES[p]} for p in paths]}


def l2n0() -> None:
    cid = "ihp_sg13g2_l2n0"
    out = ROOT / "cases" / cid
    (out / "measured").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(upstream(L2N0_GDS), out / "layout.gds")
    for label, path in L2N0_MEAS.items():
        shutil.copyfile(upstream(path), out / "measured" / f"{label}.s2p")
    write_case(out, {
        "id": cid,
        "title": "L2n0 octagonal inductor, 2 turns, 2 nH, IHP SG13G2",
        "family": "inductor",
        "domain": "onchip",
        "source": source(L2N0_GDS, *L2N0_MEAS.values()),
        "stack": "ihp_sg13g2",
        "layout": {"file": "layout.gds", "provenance": "original", "source_file": L2N0_GDS,
                   # 75/0 outlines the inductor (recognition layer)
                   "ignore_layers": [[75, 0]], "build": "importers/ihp_openpdk.py"},
        "ports": [
            {"name": "P1", "kind": "edge", "layer": "TopMetal1", "reference": "common_node",
             "segment": [[-34.2, 0.0], [-22.2, 0.0]], "z0_ohm": 50},
            {"name": "P2", "kind": "edge", "layer": "TopMetal1", "reference": "common_node",
             "segment": [[22.2, 0.0], [34.2, 0.0]], "z0_ohm": 50},
        ],
        "measurements": [{
            "label": label,
            "file": f"measured/{label}.s2p",
            "quantity": "sparams",
            "reference_plane": "inductor terminals at the end of the TopMetal1 feeds (y = 0)",
            "calibration": "probe-tip calibration, GSGSG pads and feeds removed by THRU "
                           "de-embedding with on-wafer standards",
            "instrument": f"WinCal-controlled probe station, wafer PQD701 W03, {label}",
            "source_file": path,
        } for label, path in L2N0_MEAS.items()],
        "plot": ["L(Zdiff)", "Q(Zdiff)"],
        "band": {"f_min_hz": 1.0e8, "f_max_hz": 5.0e10},
        "open_questions": [
            "Which ground reference do the two single-ended ports have after the GSGSG "
            "de-embedding? The differential quantities (Zdiff) do not depend on it.",
            "Substrate thickness and backside condition of the measured die: the IHP model "
            "uses 280 um with PEC boundaries, the L6n2 study 200 um with open boundaries.",
            "The de-embedding standards and the pad frame GDS.",
        ],
        "notes": (
            "N = 2, w = 12 um, s = 3 um, inner diameter 200 um, winding on TopMetal2 with "
            "TopMetal1 feeds and underpass. The IHP guide reports a measured SRF of 25.5 "
            "GHz and peak Q 23.5 at 9 GHz, against 22.7 GHz in its openEMS model, and "
            "attributes the gap to the conformal passivation between the turns, which its "
            "planar model did not have and the stack here does. The two dies show the "
            "die-to-die spread."),
    })
    print(f"wrote {cid}")


def line880() -> None:
    cid = "ihp_sg13g2_line880"
    out = ROOT / "cases" / cid
    (out / "measured").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(upstream(LINE_MEAS), out / "measured" / "ihp.s2p")
    w, l, wg, lg = 15.0, 880.0, 90.0, 930.0
    rect = lambda a, b: [(-a / 2, -b / 2), (a / 2, -b / 2), (a / 2, b / 2), (-a / 2, b / 2)]
    write_gds(out / "layout.gds", cid, {TOPMETAL2: [rect(l, w)], METAL1: [rect(lg, wg)]})
    write_case(out, {
        "id": cid,
        "title": "Microstrip line TopMetal2 over Metal1, 880 um, IHP SG13G2",
        "family": "line",
        "domain": "onchip",
        "source": source(LINE_MEAS),
        "stack": "ihp_sg13g2",
        "layout": {"file": "layout.gds", "provenance": "reconstructed",
                   "build": "importers/ihp_openpdk.py"},
        "ports": [
            {"name": "P1", "kind": "vertical", "from": "Metal1", "to": "TopMetal2",
             "segment": [[-l / 2, -w / 2], [-l / 2, w / 2]], "z0_ohm": 50},
            {"name": "P2", "kind": "vertical", "from": "Metal1", "to": "TopMetal2",
             "segment": [[l / 2, -w / 2], [l / 2, w / 2]], "z0_ohm": 50},
        ],
        "measurements": [{
            "label": "ihp",
            "file": "measured/ihp.s2p",
            "quantity": "sparams",
            "reference_plane": "line ends, pad sections de-embedded",
            "calibration": "probe-tip calibration, pad sections de-embedded from the raw "
                           "measurement (method not documented)",
            "instrument": "WinCal-controlled probe station, measured 2014-04-30",
            "source_file": LINE_MEAS,
        }],
        "band": {"f_min_hz": 5.0e8, "f_max_hz": 1.1e11},
        "open_questions": [
            "The de-embedding method behind 'corrected' and the GDS of the test structure "
            "with pads; the geometry here is the one of IHP's openEMS model.",
        ],
        "notes": (
            "Line 15 um wide on TopMetal2, 880 um long, over a Metal1 ground strip 90 um "
            "wide and 930 um long, as in the IHP model. The ground strip shields the "
            "substrate, so the uncertain substrate thickness of the stack matters little "
            "here. A 2580 um line of the same kind was measured as well, but only plots "
            "of it are public."),
    })
    print(f"wrote {cid}")


if __name__ == "__main__":
    l2n0()
    line880()
