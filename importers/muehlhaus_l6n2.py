"""Muehlhaus / IHP, L6n2 spiral inductor on SG13G2, measured to 50 GHz.

Source: https://github.com/VolkerMuehlhaus/gds2palace_ihp_sg13g2 (GPL-3.0-or-later),
more_examples/measured_vs_simulated/more_accurate_models_L6n2_v2. The GDS and the
de-embedded measurement are copied byte for byte; the stack is transcribed by hand
into stacks/ihp_sg13g2_200um.yaml.

    python importers/muehlhaus_l6n2.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emref.importing import github, write_case  # noqa: E402
from emref.stack import ROOT  # noqa: E402

REPO = "VolkerMuehlhaus/gds2palace_ihp_sg13g2"
COMMIT = "101e119fc08469fa3547eface93e1edc2dddfad7"
DIR = "more_examples/measured_vs_simulated/more_accurate_models_L6n2_v2"
GDS = f"{DIR}/L6n2_with_ports.gds"
MEAS = f"{DIR}/meas_L5_6n2_THRU_deemb.S2P"
FILES = {
    GDS: "30dd46f38e30a7cf1677076be2a0b64049bdadbe9da82d7235891481555f3ddd",
    MEAS: "cbfb4cfcd777ca9f958de4f1db7d43caeee2ab18dc9523f8b9350fa3ecc274cd",
}
CID = "ihp_sg13g2_l6n2"


def upstream(path: str) -> Path:
    return github(REPO, COMMIT, path, FILES[path])


def main() -> None:
    out = ROOT / "cases" / CID
    (out / "measured").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(upstream(GDS), out / "layout.gds")
    shutil.copyfile(upstream(MEAS), out / "measured" / "ihp.s2p")
    write_case(out, {
        "id": CID,
        "title": "L6n2 octagonal spiral inductor, 4 turns, IHP SG13G2",
        "family": "inductor",
        "domain": "onchip",
        "source": {
            "title": "gds2palace_ihp_sg13g2, measured vs simulated, L6n2 v2",
            "authors": ["Volker Muehlhaus", "IHP"],
            "url": f"https://github.com/{REPO}/tree/main/{DIR}",
            "commit": COMMIT,
            "license": "GPL-3.0-or-later",
            "files": [{"path": p, "sha256": h} for p, h in FILES.items()],
        },
        "stack": "ihp_sg13g2",
        "layout": {
            "file": "layout.gds",
            "cell": "L_6n2",
            "provenance": "original",
            "source_file": GDS,
            # 27/0 is the IND recognition marker; 201/0 and 202/0 carry the port
            # sheets of the source, restated as port segments.
            "ignore_layers": [[27, 0], [201, 0], [202, 0]],
            "build": "importers/muehlhaus_l6n2.py",
        },
        "ports": [
            {"name": "P1", "kind": "vertical", "from": "SUBGND", "to": "TopMetal1",
             "segment": [[-103.675, -50.0], [-95.675, -50.0]], "z0_ohm": 50},
            {"name": "P2", "kind": "vertical", "from": "SUBGND", "to": "TopMetal1",
             "segment": [[-47.305, -50.0], [-39.305, -50.0]], "z0_ohm": 50},
        ],
        "measurements": [{
            "label": "ihp",
            "file": "measured/ihp.s2p",
            "quantity": "sparams",
            "reference_plane": "lead ends at y = -50 um, assuming the de-embedding removes "
                               "pads and feeds up to there",
            "calibration": "probe-tip calibration, then THRU-dummy de-embedding "
                           "(method not documented)",
            "instrument": "WinCal-controlled probe station, die PQD701W03 x2y7",
            "source_file": MEAS,
        }],
        "plot": ["L(Zdiff)", "Q(Zdiff)"],
        "band": {"f_min_hz": 1.0e8, "f_max_hz": 5.0e10},
        "open_questions": [
            "Which de-embedding was applied (THRU dummy only, open-short, other), and where "
            "exactly does it put the reference plane relative to this GDS?",
            "The full test structure GDS (pads, feeds, ground ring, fill) and the dummy "
            "structures used for de-embedding.",
            "Was the die measured on a metal chuck (then the bottom boundary is PEC at the "
            "backside) and is the final thickness 200 um?",
            "Which physical terminal is port 1 of the measurement?",
        ],
        "notes": (
            "The upstream study reports L = 5.01 nH, peak Q 15.96 at 4.71 GHz and a "
            "measured self-resonance of 11.07 GHz, and reaches agreement in SRF within "
            "0.1 to 1 percent only with the conformal oxide over TopMetal2 that the stack "
            "carries."),
    })
    print(f"wrote {CID}")


if __name__ == "__main__":
    main()
