"""Hatab, SMA-to-microstrip multiline TRL kit on 1.6 mm FR4, 0.1 to 14 GHz.

Source: https://github.com/ZiadHatab/SMA-PCB-mTRL-kit (MIT). Two boards from one
order: a multiline TRL board (50 Ohm lines of 0, 2.5, 10, 15 and 50 mm relative
length, a symmetric short) and a DUT board with identical SMA launches. The
measured DUT is the 90 Ohm stepped line.

Reference planes. The thru is 50 mm from board edge to board edge, so the
calibration puts each plane 25 mm in from the edge. On the DUT board the 90 Ohm
section runs from 25 mm to 45 mm: the planes sit exactly on the two impedance
steps. The layout keeps 2 mm of the 50 Ohm line in front of each step and the
ports de-embed them.

Stack. The source only says "JLCPCB standard 2-layer 1.6 mm FR4". Permittivity and
loss tangent are therefore fitted to the propagation constant that the multiline
TRL extracts from the calibration lines (Djordjevic-Sarkar, Hammerstad-Jensen with
Kirschning-Jansen dispersion, smooth copper). Those lines are separate structures
from the DUT, so the comparison against the DUT is not circular, but the values
are reconstructed, not stated.

    python importers/hatab_sma_fr4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import skrf
import yaml
from scipy.optimize import least_squares
from shapely.affinity import affine_transform
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emref.importing import (github, gerber_copper, write_case, write_gds,  # noqa: E402
                             write_touchstone)
from emref.stack import ROOT  # noqa: E402

REPO = "ZiadHatab/SMA-PCB-mTRL-kit"
COMMIT = "031c6947e991b8d8807231ca0fbf627c27d32279"
FILES = {
    "CAD files/Gerber/DUT/duts_copper_signal_top.gbr": "f1b9411887fff0a30fa6df578d2b561f1c19148e9c471602023ba6e9ee036ebe",
    "Measurements/line_0_0mm.s2p": "534603a2debee04d5f4f8032e165e0dfaddb3d9c0709b4c4b703a0df1e1cfb71",
    "Measurements/line_2_5mm.s2p": "b46277fac7c2e3b51c64ef13ee255f3ad7552ba8ff0dec4736d267867db07fab",
    "Measurements/line_10_0mm.s2p": "57c83c731b16ab8d6ab538312049f2f4154de445fe4c98aa1178a1bb32ea4c7a",
    "Measurements/line_15_0mm.s2p": "6aac01d0debf4e2c87c959aaee68e43e3a0e7f30f1d3875c0490c11175a6e604",
    "Measurements/line_50_0mm.s2p": "6f8363e4b8b40013e363f491d757cf28d1302a2a3483fe3db0eeba7e49c6c666",
    "Measurements/short_0_0mm.s2p": "31696d43de40c51b7cfdaa9d7e633bb7bc2a35a5bf05e620318c711afe89d105",
    "Measurements/Gamma_21.s1p": "eed2b6fe4bfd196661f94ddd881e83e39d9cecf81f8df31c534f8a91ba1543b9",
    "Measurements/Gamma_12.s1p": "b9e4d0c4c94befc2c193b5f0d516b51eab4b4161ff5fc7df3de5d6372174fec1",
    "Measurements/step_line.s2p": "08039c5dc94bf5e4eac544734ec05c69dd4777b01287d9d40053e8c846373b18",
}
BAND = "0.1ghz-14ghz"           # the source finds the lines stable up to 14 GHz
LINES_MM = [0, 2.5, 10, 15, 50]
CID = "hatab_sma_fr4_step90"

# Board geometry, mm, from the Gerber: the 90 Ohm line on the DUT board runs along
# y at x = 85, its section from y = 25 to 45; reference planes at y = 25 and 45.
X_CENTRE, Y_PLANE1, Y_PLANE2, FEED = 85.0, 25.0, 45.0, 2.0
W50 = 2.941
H_UM, T_UM = 1520.0, 35.0       # FR4 core and 1 oz copper, see the stack notes


def upstream(path: str) -> Path:
    return github(REPO, COMMIT, path, FILES[path])


def net(name: str) -> skrf.Network:
    return skrf.Network(str(upstream(f"Measurements/{name}")))[BAND]


def calibration() -> skrf.calibration.TUGMultilineTRL:
    lines = [net(f"line_{x:.1f}mm.s2p".replace(".", "_", 1)) for x in LINES_MM]
    cal = skrf.calibration.TUGMultilineTRL(
        line_meas=lines, line_lengths=[x * 1e-3 for x in LINES_MM], er_est=3.5,
        reflect_meas=net("short_0_0mm.s2p"), reflect_est=-1, reflect_offset=0,
        switch_terms=(net("Gamma_21.s1p"), net("Gamma_12.s1p")))
    cal.run()
    return cal


def fit_dielectric(cal) -> tuple[float, float]:
    """er and tand at 1 GHz such that a Hammerstad-Jensen microstrip with
    Djordjevic-Sarkar dielectric reproduces the measured propagation constant of
    the 50 Ohm line. Below 0.5 GHz the short lines carry too little phase."""
    from skrf.media import MLine
    freq, g = cal.frequency, cal.gamma
    k = freq.f >= 0.5e9

    def residual(p):
        m = MLine(frequency=freq, w=W50 * 1e-3, h=H_UM * 1e-6, t=T_UM * 1e-6, ep_r=p[0],
                  tand=p[1], rho=1 / 58e6, rough=0, model="hammerstadjensen",
                  disp="kirschningjansen", diel="djordjevicsvensson", f_epr_tand=1e9)
        gm = m.gamma
        return np.concatenate([(gm.imag - g.imag)[k] / g.imag[k],
                               10 * (gm.real - g.real)[k] / g.imag[k]])

    r = least_squares(residual, [4.4, 0.02], bounds=([2, 0], [6, 0.1]))
    return float(round(r.x[0], 3)), float(round(r.x[1], 4))


def layout() -> tuple[list, float]:
    """The DUT island clipped to the planes plus the 50 Ohm feeds, rotated so the
    line runs along +x with plane 1 at x = FEED. Returns polygon (um) and length."""
    copper = gerber_copper(upstream("CAD files/Gerber/DUT/duts_copper_signal_top.gbr"))
    clip = box(X_CENTRE - 5, Y_PLANE1 - FEED, X_CENTRE + 5, Y_PLANE2 + FEED)
    part = copper.intersection(clip)
    # (x, y) mm -> (y - y0, -(x - x_c)) um: a rotation, the line along +x
    y0 = Y_PLANE1 - FEED
    part = affine_transform(part, [0, 1000, -1000, 0, -1000 * y0, 1000 * X_CENTRE])
    pts = [(round(x, 3), round(y, 3)) for x, y in part.exterior.coords[:-1]]
    return pts, 1000 * (Y_PLANE2 - Y_PLANE1 + 2 * FEED)


def main() -> None:
    cal = calibration()
    er, tand = fit_dielectric(cal)
    out = ROOT / "cases" / CID
    poly, total = layout()
    write_gds(out / "layout.gds", CID, {(1, 0): [poly]})
    dut = cal.apply_cal(net("step_line.s2p"))
    write_touchstone(out / "measured" / "tug.s2p", dut, [
        f"{CID}: multiline TRL calibrated (TUGMultilineTRL, scikit-rf {skrf.__version__})",
        f"raw data {REPO} @ {COMMIT[:7]}, switch terms corrected",
        "reference impedance: characteristic impedance of the 2.941 mm 50 Ohm line,",
        "NOT 50 Ohm; the R value below is a placeholder required by the format"])

    stack = {
        "name": "hatab_sma_fr4",
        "title": "JLCPCB standard 2-layer FR4, 1.6 mm, bare top copper over a ground plane",
        "source": {"authors": ["Ziad Hatab"], "url": f"https://github.com/{REPO}",
                   "commit": COMMIT, "license": "MIT"},
        "boundaries": {"bottom": {"sigma_s_per_m": 58.0e6, "provenance": "assumed"},
                       "top": "open"},
        "lateral": "infinite",
        "dielectrics": [{
            "name": "FR4 core", "thickness_um": H_UM, "er": er, "tand": tand,
            "model": "djordjevic_sarkar", "f_ref_hz": 1.0e9, "f_low_hz": 1.0e3,
            "f_high_hz": 1.0e12,
            "provenance": {"default": "reconstructed", "thickness_um": "assumed"}}],
        "conductors": [{
            "name": "TOP", "kind": "metal", "gds": [1, 0],
            "z_um": [H_UM, H_UM + T_UM], "sigma_s_per_m": 58.0e6,
            "provenance": {"default": "datasheet", "sigma_s_per_m": "assumed"}}],
        "notes": (
            f"er {er} and tand {tand} at 1 GHz are fitted to the propagation constant of "
            "the calibration lines (0.5 to 14 GHz) with a Djordjevic-Sarkar dielectric "
            "and smooth copper, so tand also absorbs the excess conductor loss of the "
            "rough foil. The fit is insensitive to the core thickness: 1.50 to 1.55 mm "
            "changes er by less than 0.01. 1.52 mm is the core of a 1.6 mm finished "
            "board, 35 um the 1 oz outer copper of the board house. The lines are not "
            "covered by solder mask; the surface finish of the copper is not stated."),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "stack.yaml").write_text(yaml.safe_dump(stack, sort_keys=False, width=100),
                                    encoding="utf-8", newline="\n")

    w = W50 * 1000 / 2
    write_case(out, {
        "id": CID,
        "title": "Stepped-impedance microstrip 50-90-50 Ohm, 20 mm section, FR4",
        "family": "stepped_line",
        "domain": "pcb",
        "source": {
            "title": "FR4 microstrip multiline TRL board",
            "authors": ["Ziad Hatab"],
            "url": f"https://github.com/{REPO}",
            "commit": COMMIT,
            "license": "MIT",
            "files": [{"path": p, "sha256": h} for p, h in FILES.items()],
        },
        "stack": "stack.yaml",
        "layout": {"file": "layout.gds", "provenance": "converted",
                   "build": "importers/hatab_sma_fr4.py"},
        "ports": [
            {"name": "P1", "kind": "line", "layer": "TOP", "reference": "bottom_boundary",
             "segment": [[0.0, -w], [0.0, w]], "deembed_um": FEED * 1000, "z0_ohm": "line"},
            {"name": "P2", "kind": "line", "layer": "TOP", "reference": "bottom_boundary",
             "segment": [[total, w], [total, -w]], "deembed_um": FEED * 1000,
             "z0_ohm": "line"},
        ],
        "measurements": [{
            "label": "tug",
            "file": "measured/tug.s2p",
            "quantity": "sparams",
            "reference_plane": "on the two impedance steps, 25 mm from each board edge",
            "calibration": "multiline TRL on a separate board of the same order with "
                           "identical SMA launches: five 50 Ohm lines (0 to 50 mm) and a "
                           "symmetric short; switch terms measured",
            "instrument": "Rohde & Schwarz ZVA67, SMA (Johnson 142-0701-801)",
            "derive": "importers/hatab_sma_fr4.py",
        }],
        "band": {"f_min_hz": float(dut.f[0]), "f_max_hz": float(dut.f[-1])},
        "notes": (
            "The source measured to 20 GHz but finds the lines usable only to 14 GHz, "
            "the band kept here. The DUT and the calibration standards sit on different "
            "boards, so board-to-board variation of the launches and of the substrate "
            "enters the result. S-parameters are normalised to the characteristic "
            "impedance of the 50 Ohm line."),
    })
    print(f"wrote {CID}: er {er}, tand {tand}")


if __name__ == "__main__":
    main()
