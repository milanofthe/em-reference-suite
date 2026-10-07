"""TerayTech RF Works, microstrip filters on an RO4350B-class hybrid board.

Source: https://github.com/TerayTech/microstrip_filters (MIT). Three measured
filters: a 5 GHz and an 8 GHz radial-stub low-pass, and a 10 GHz coupled-line
band-pass in a via-fenced ground frame. Gerber files per board, S-parameters from a
Keysight E5080B with the calibration at the SMA connectors.

These cases are incomplete by construction: the measurement includes the two SMA
connectors, which are not part of the layout. They are kept because the filter
geometry and the data are public and clean, and because a connector model or a
de-embedding of it would make them complete.

Geometry: top copper from the GTL layer, plated holes from the crosses of the
Altium drill guide (GG1), which are drawn at hole size. Layers 2, 3 and the bottom
are solid planes; layer 2 is the reference under the 0.254 mm RF laminate.

    python importers/teraytech_filters.py
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import yaml
from shapely.geometry import MultiPolygon, Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emref.importing import (circle, github, gerber_copper, gerber_drill_guide,  # noqa: E402
                             write_case, write_gds)
from emref.stack import ROOT  # noqa: E402

REPO = "TerayTech/microstrip_filters"
COMMIT = "1ae960799e0048960539dba38e34655237a50236"
FILES = {
    "PCB_GERBER_FAB/GERBER_5G_LPF.zip": "7f015978a311808ca9ccec42a3560cc131af91c48ec326db5ae7290f9d2255ce",
    "PCB_GERBER_FAB/GERBER_8G_LPF.zip": "d93c64f22c010ea16c7ea15e6d519da787d6fbf80b4288483fc6d33e8b54ce72",
    "PCB_GERBER_FAB/GERBER_10GHz_BPF.zip": "04890669a3aac71360294d2f0fd6740ebde8f79a1c5ac180db537beb00d8d026",
    "MEAS_RESULT/LPF1.s2p": "5d2eaeb6bc5642519e830a36077e7120af93b3ba7687ff1ae877aa204acd1d20",
    "MEAS_RESULT/LPF2.s2p": "0876c1ba9d79c444d8901d8992f87bc5c3308cf61d24d65d9c8471d56dd6f227",
    "MEAS_RESULT/10GBPF.s2p": "9f476962f45eeef3798f57916dff7e384630830b9c51de18fb20856baaded306",
}
BOARDS = [
    # case id, title, Gerber zip, file stem inside, measurement
    ("teraytech_lpf_5g", "Radial-stub microstrip low-pass filter, 5 GHz, RO4350B",
     "PCB_GERBER_FAB/GERBER_5G_LPF.zip", "FILTERS", "MEAS_RESULT/LPF1.s2p"),
    ("teraytech_lpf_8g", "Radial-stub microstrip low-pass filter, 8 GHz, RO4350B",
     "PCB_GERBER_FAB/GERBER_8G_LPF.zip", "FILTERS2", "MEAS_RESULT/LPF2.s2p"),
    ("teraytech_bpf_10g", "Coupled-line band-pass filter, 10 GHz, RO4350B",
     "PCB_GERBER_FAB/GERBER_10GHz_BPF.zip", "10GHz_BPF", "MEAS_RESULT/10GBPF.s2p"),
]
TOP, VIA, OUTLINE = (1, 0), (2, 0), (100, 0)
H_UM, T_UM = 254.0, 35.0


def upstream(path: str) -> Path:
    return github(REPO, COMMIT, path, FILES[path])


def gerbers(zip_path: str, stem: str) -> dict[str, Path]:
    archive = upstream(zip_path)
    folder = archive.with_suffix("")
    if not folder.exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(folder)
    return {p.suffix[1:].upper(): p for p in folder.rglob(f"{stem}.*")}


def board(files: dict[str, Path]):
    """Top copper islands, holes and the outline rectangle, in mm."""
    copper = gerber_copper(files["GTL"])
    islands = list(copper.geoms) if isinstance(copper, MultiPolygon) else [copper]
    holes = gerber_drill_guide(files["GG1"])
    x0, y0, x1, y1 = gerber_copper(files["GKO"]).bounds
    w = 0.1                                     # outline is drawn as a 0.1 mm track
    return islands, holes, (x0 + w / 2, y0 + w / 2, x1 - w / 2, y1 - w / 2)


def ports(islands, holes, rect) -> list[list]:
    """The signal trace ends at the left and right board edge: islands that reach
    the edge and carry no plated hole (the ground pads do)."""
    x0, _, x1, _ = rect
    out = []
    for edge in (x0, x1):
        for isl in islands:
            bx0, _, bx1, _ = isl.bounds
            if min(abs(bx0 - edge), abs(bx1 - edge)) > 0.3:
                continue
            if any(isl.contains(Point(x, y)) for x, y, _ in holes):
                continue
            xs = bx0 if abs(bx0 - edge) < abs(bx1 - edge) else bx1
            ys = [y for x, y in isl.exterior.coords if abs(x - xs) < 1e-6]
            out.append([[xs, min(ys)], [xs, max(ys)]])
    if len(out) != 2:
        raise RuntimeError(f"found {len(out)} signal terminals, expected 2")
    return out


def stack() -> dict:
    return {
        "name": "teraytech_ro4350b",
        "title": "RO4350B-class RF laminate 0.254 mm over a ground plane, 4-layer hybrid",
        "source": {"authors": ["TerayTech RF Works"], "url": f"https://github.com/{REPO}",
                   "commit": COMMIT, "license": "MIT"},
        "boundaries": {"bottom": {"sigma_s_per_m": 58.0e6, "provenance": "assumed"},
                       "top": "open"},
        "lateral": "outline",
        "dielectrics": [{
            "name": "RO4350B-class laminate", "thickness_um": H_UM, "er": 3.66,
            "tand": 0.0037, "provenance": {"default": "datasheet", "tand": "assumed"}}],
        "conductors": [
            {"name": "TOP", "kind": "metal", "gds": list(TOP), "z_um": [H_UM, H_UM + T_UM],
             "sigma_s_per_m": 58.0e6,
             "provenance": {"default": "datasheet", "sigma_s_per_m": "assumed"}},
            {"name": "VIA", "kind": "via", "gds": list(VIA), "z_um": [0.0, H_UM],
             "sigma_s_per_m": 58.0e6, "provenance": "assumed"},
        ],
        "notes": (
            "From FAB_DOC.md of the source: RO4350B for the RF layer (Dk 3.66, Df "
            "0.0037, 0.254 mm), FR-4 below, 1 oz ED copper, OSP finish, no solder mask "
            "over the circuits. The boards were built on a Chinese RO4350B variant with "
            "the same Dk and a lower, unstated Df, so tand is an upper bound. The layer "
            "thicknesses are described by the source as approximated to a 0.8 mm board. "
            "Plated holes are modelled as solid copper cylinders from the ground plane "
            "to the top copper."),
    }


def main() -> None:
    for cid, title, zip_path, stem, meas in BOARDS:
        out = ROOT / "cases" / cid
        islands, holes, rect = board(gerbers(zip_path, stem))
        x0, y0, x1, y1 = rect
        um = lambda pts: [(round((x - x0) * 1000, 3), round((y - y0) * 1000, 3)) for x, y in pts]
        copper = [um(i.exterior.coords[:-1]) for i in islands]
        vias = [um(circle(x, y, d)) for x, y, d in holes]
        outline = [um([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])]
        write_gds(out / "layout.gds", cid, {TOP: copper, VIA: vias, OUTLINE: outline})
        (out / "measured").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(upstream(meas), out / "measured" / "keysight.s2p")
        (out / "stack.yaml").write_text(yaml.safe_dump(stack(), sort_keys=False, width=100),
                                        encoding="utf-8", newline="\n")
        sizes = sorted({round(d, 3) for _, _, d in holes})
        p1, p2 = (um(seg) for seg in ports(islands, holes, rect))
        write_case(out, {
            "id": cid,
            "title": title,
            "family": "filter",
            "domain": "pcb",
            "source": {
                "title": "Microstrip Filters",
                "authors": ["TerayTech RF Works"],
                "url": f"https://github.com/{REPO}",
                "commit": COMMIT,
                "license": "MIT",
                "files": [{"path": p, "sha256": FILES[p]} for p in (zip_path, meas)],
            },
            "stack": "stack.yaml",
            "layout": {"file": "layout.gds", "provenance": "converted",
                       "outline_layer": list(OUTLINE), "build": "importers/teraytech_filters.py"},
            "ports": [
                {"name": "P1", "kind": "edge", "layer": "TOP", "reference": "bottom_boundary",
                 "segment": [list(p1[0]), list(p1[1])], "z0_ohm": 50},
                {"name": "P2", "kind": "edge", "layer": "TOP", "reference": "bottom_boundary",
                 "segment": [list(p2[0]), list(p2[1])], "z0_ohm": 50},
            ],
            "measurements": [{
                "label": "keysight",
                "file": "measured/keysight.s2p",
                "quantity": "sparams",
                "reference_plane": "coaxial, at the SMA connector interfaces; the "
                                   "connectors and their launch are part of the data",
                "calibration": "2-port coaxial calibration of the VNA (type not stated)",
                "instrument": "Keysight E5080B",
                "source_file": meas,
            }],
            "band": {"f_min_hz": 1.0e7, "f_max_hz": 1.8e10},
            "open_questions": [
                "The data include the SMA end-launch connectors (SMA-KHD23), which are not "
                "part of the layout. A connector model or a measurement of the connector "
                "pair (thru) is needed to move the reference plane onto the board.",
                f"Drill diameters are read from the drill guide crosses ({sizes} mm); the "
                "source has no NC drill file. Whether the large holes are plated is not "
                "stated.",
                "Actual loss tangent of the RO4350B-class laminate and the true layer "
                "thicknesses of the hybrid stack.",
            ],
            "notes": (
                "Ports are placed where the signal trace meets the board edge, as edge "
                "ports to the ground plane under the laminate. The ground pads beside the "
                "trace at both edges are the launch pads of the connectors."),
        })
        print(f"wrote {cid}: {len(islands)} copper islands, {len(holes)} holes {sizes} mm")


if __name__ == "__main__":
    main()
