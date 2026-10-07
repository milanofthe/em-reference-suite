"""Hatab et al., stepped-impedance microstrip lines on Megtron 7, 1 to 150 GHz.

Source: Z. Hatab, M. E. Gadringer, A. B. Alothman Alterkawi, W. Boesch, "Validation
of the Reference Impedance in Multiline Calibration with Stepped Impedance
Standards", IEEE OJIM vol. 2, 2023, doi:10.1109/OJIM.2023.3315349, and the
measurement repository linked from it.

The board carries two multiline TRL kits. The primary one is a set of 50 Ohm lines
(trace 94 um); its calibration puts the reference planes at the middle of the thru
and normalises to the characteristic impedance of that line. Measured against it,
each standard of the second kit is a stepped line between the two planes:

    0.5 mm of 50 Ohm | (1.0 + X) mm of 30 Ohm | 0.5 mm of 50 Ohm

with X the line length of that standard relative to its thru (main.py of the
source: d1 = d2 = 0.5 mm offsets on either side of the second thru's centre). The
pads and feeds lie outside the planes and drop out. Those six stepped lines are the
cases written here.

    python importers/hatab_megtron7.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import skrf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emref.importing import fetch, write_case, write_gds, write_touchstone  # noqa: E402
from emref.stack import ROOT  # noqa: E402

REPO = "ZiadHatab/verification-multiline-trl-calibration"
COMMIT = "021e7359b77c33507a407f6b35205dbe9152e467"
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/Measurements"
ZIPS = {
    "line_50__0_0mm": "f17622b441865d461f2a6f9b25f265105a3a063df4c87e3b0890e76ebb2450cb",
    "line_50__0_5mm": "0d658d25fbf9649b78808e5751f2d760c254c88b4e12ac107b6c4615d7249d84",
    "line_50__1_0mm": "55066dfba6bf4757821f865e3623d675965709d5a8362071b5e9cc3b501f505f",
    "line_50__1_5mm": "6076ae2d00f3beb909fbe1f850247d33456e9a27346088f0d58fcc7cf1b1a1ba",
    "line_50__2_0mm": "67d6ac863be5f4efe8656b522f452c2022763ba65745726796bcb2d326d8ed05",
    "line_50__3_0mm": "00955778918c44c6a25a7b88aa2545e1b4da9b8256fcb3d7164a214801406faf",
    "line_50__5_0mm": "022e319223a80d3afbb0a32eeffebfbb311aa542233a412bb327e1903a8c8c39",
    "line_50__6_5mm": "ddcaf1cfef9a89acb3f8eef03d549bfdcd9922b26f6a37401fb00cc55bc80c51",
    "line_30__0_0mm": "78341eb67fb982f3f4b721b153b0fbff09405dc314d97c15d8013cf8f4d5696f",
    "line_30__0_5mm": "59b2868ca8fef873f613a58908cab1da0458fb38bd36817ec198431cd0d61a1e",
    "line_30__1_0mm": "578e7679f8e789aa479433013a47544fbc8653f8c7d415e4c31ce6f6a391ebd9",
    "line_30__3_0mm": "38b4b5ef487dda1efefbda735b56b161770f32eff654455ebc98d6651f085725",
    "line_30__5_0mm": "579945d5a9414822a9295eccab5df521a1c25ee1cb4f77276c8fd56133001a78",
    "line_30__6_5mm": "261ffc7cfbb655d26b691dc56ed9d5430f5fe6a15f3f9e565a2abd07e0dcee60",
    "open": "249352b7a30958ea6645c62f92db5c4a5045952678c086b21addcfdf6acb7969",
}

# Geometry, OJIM paper Table I (cross-section measurements), um.
W50, W30 = 94.0, 209.0
OFFSET = 500.0                  # 50 Ohm stub on either side of the step, d1 = d2
LINES_MM = [0, 0.5, 1, 1.5, 2, 3, 5, 6.5]
STEPS_MM = [0, 0.5, 1, 3, 5, 6.5]
TOP = (1, 0)                    # GDS layer of the trace, see stacks/hatab_megtron7.yaml


def _tag(x_mm: float) -> str:
    """Upstream file naming: 0.5 mm -> '0_5', 3 mm -> '3_0'."""
    return f"{x_mm:.1f}".replace(".", "_")


def measured(name: str) -> skrf.Network:
    """Mean of the repeated sweeps. Each sweep is stored as two wave-parameter
    files A and B with S = B A^-1; the source already corrected the switch terms."""
    archive = fetch(f"{RAW}/{name}.zip", ZIPS[name])
    folder = archive.with_suffix("")
    if not folder.exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(folder)
    files = sorted(folder.rglob("*.s2p"))
    a = [skrf.Network(str(f)) for f in files if f.stem.startswith(f"{name}_A")]
    b = [skrf.Network(str(f)) for f in files if f.stem.startswith(f"{name}_B")]
    assert a and len(a) == len(b), f"{name}: {len(a)} A and {len(b)} B sweeps"
    s = np.mean([nb.s @ np.linalg.inv(na.s) for na, nb in zip(a, b)], axis=0)
    return skrf.Network(frequency=a[0].frequency, s=s, name=name)


def calibration() -> skrf.calibration.TUGMultilineTRL:
    lines = [measured(f"line_50__{_tag(x)}mm") for x in LINES_MM]
    cal = skrf.calibration.TUGMultilineTRL(
        line_meas=lines, line_lengths=[x * 1e-3 for x in LINES_MM], er_est=2.5 - 1e-5j,
        reflect_meas=measured("open"), reflect_est=1, reflect_offset=-5.3e-3 / 2)
    cal.run()
    return cal


def stepped_line(x_mm: float) -> tuple[list, float]:
    """Outline of 50 | 30 | 50 Ohm, reference plane 1 at x = 0. Returns the polygon
    and the total length."""
    l30 = 1000.0 + 1000.0 * x_mm
    total = 2 * OFFSET + l30
    a, b = W50 / 2, W30 / 2
    pts = [(0, -a), (OFFSET, -a), (OFFSET, -b), (OFFSET + l30, -b), (OFFSET + l30, -a),
           (total, -a), (total, a), (OFFSET + l30, a), (OFFSET + l30, b), (OFFSET, b),
           (OFFSET, a), (0, a)]
    return pts, total


def main() -> None:
    cal = calibration()
    for x in STEPS_MM:
        l30 = 1.0 + x
        cid = f"hatab_megtron7_step30_{l30:.1f}mm".replace(".", "p")
        out = ROOT / "cases" / cid
        poly, total = stepped_line(x)
        write_gds(out / "layout.gds", cid[:31], {TOP: [poly]})
        dut = cal.apply_cal(measured(f"line_30__{_tag(x)}mm"))
        write_touchstone(out / "measured" / "tug.s2p", dut, [
            f"{cid}: multiline TRL calibrated (TUGMultilineTRL, scikit-rf {skrf.__version__})",
            f"raw data {REPO} @ {COMMIT[:7]}, mean of the repeated sweeps",
            "reference impedance: characteristic impedance of the 94 um 50 Ohm line,",
            "NOT 50 Ohm; the R value below is a placeholder required by the format"])
        write_case(out, {
            "id": cid,
            "title": f"Stepped-impedance microstrip 50-30-50 Ohm, {l30:g} mm section, Megtron 7",
            "family": "stepped_line",
            "domain": "pcb",
            "source": {
                "title": "Validation of the Reference Impedance in Multiline Calibration "
                         "with Stepped Impedance Standards",
                "authors": ["Ziad Hatab", "Michael Ernst Gadringer",
                            "Ahmad Bader Alothman Alterkawi", "Wolfgang Boesch"],
                "url": f"https://github.com/{REPO}",
                "doi": "10.1109/OJIM.2023.3315349",
                "commit": COMMIT,
                "license": "BSD-3-Clause",
                "files": [{"path": f"Measurements/{n}.zip", "sha256": h}
                          for n, h in ZIPS.items()],
                "citation": "Z. Hatab et al., IEEE Open Journal of Instrumentation and "
                            "Measurement, vol. 2, pp. 1-12, 2023",
            },
            "stack": "hatab_megtron7",
            "layout": {"file": "layout.gds", "provenance": "reconstructed",
                       "build": "importers/hatab_megtron7.py"},
            "ports": [
                {"name": "P1", "kind": "line", "layer": "TOP", "reference": "bottom_boundary",
                 "segment": [[0.0, -W50 / 2], [0.0, W50 / 2]], "z0_ohm": "line"},
                {"name": "P2", "kind": "line", "layer": "TOP", "reference": "bottom_boundary",
                 "segment": [[total, -W50 / 2], [total, W50 / 2]], "z0_ohm": "line"},
            ],
            "measurements": [{
                "label": "tug",
                "file": "measured/tug.s2p",
                "quantity": "sparams",
                "reference_plane": "P1 and P2: middle of the thru of the 50 Ohm multiline "
                                   "TRL kit, 0.5 mm before each impedance step",
                "calibration": "GSG probes (150 um pitch), multiline TRL with eight 50 Ohm "
                               "lines (0 to 6.5 mm) and an offset open on the same board",
                "instrument": "Anritsu VectorStar, 25 sweeps, IF bandwidth 100 Hz",
                "derive": "importers/hatab_megtron7.py",
            }],
            "band": {"f_min_hz": float(dut.f[0]), "f_max_hz": float(dut.f[-1])},
            "notes": (
                "The calibration uses all eight 50 Ohm lines of the board, the source paper "
                "used six of them. Trace widths are 94 um and 209 um, each +-15 um from "
                "cross-sections; along the line the width varies within that tolerance. "
                "The S-parameters are normalised to the characteristic impedance of the "
                "94 um line, so a solver result has to be normalised the same way "
                "(de-embedded wave port on the 50 Ohm line, or renormalised to its "
                "computed Z0). The calibrated data are reciprocal to 0.003 below 5 GHz, "
                "but |S12 - S21| grows to about 0.1 at 150 GHz with |S12/S21| = 1, a phase "
                "difference. The source's own multiline TRL code gives the identical "
                "result, so this is a property of the measurement (probe contact "
                "repeatability between standards and DUT is the likely cause) and a measure "
                "of its uncertainty at the top of the band."),
        })
        print(f"wrote {cid}")


if __name__ == "__main__":
    main()
