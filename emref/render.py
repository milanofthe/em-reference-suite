"""The two README figures of a case: the layout seen from the top, and the
measured S-parameters (with contributed solver results on top, once there are any).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import style
from .case import Case


def layout(case: Case, path: Path) -> Path:
    style.setup()
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    stack = case.stack
    layers, _ = case.geometry()
    order = sorted(layers, key=lambda n: stack.z(n)[0])
    metals = [n for n in order if stack.conductor(n)["kind"] == "metal"]

    fig, ax = plt.subplots(figsize=style.FIG_SIZE)
    for name in order:
        if stack.conductor(name)["kind"] == "via":
            face, alpha = style.INK, 0.9
        else:
            face, alpha = style.SERIES[metals.index(name) % len(style.SERIES)], 0.85
        ax.add_collection(PolyCollection(layers[name], facecolors=face, edgecolors=face,
                                         linewidths=0.3, alpha=alpha, label=name))
    for p in case.ports():
        (ax_, ay), (bx, by) = p.segment
        ax.plot([ax_, bx], [ay, by], color=style.INK, lw=3, solid_capstyle="butt", zorder=5)
        ax.plot(*p.midpoint, marker="o", ms=4, color=style.INK, zorder=5)
        ax.annotate(p.name, p.midpoint, textcoords="offset points", xytext=(5, -10),
                    color=style.INK, fontsize=style.FONT_SIZE)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.06)
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=style.FONT_SIZE - 1)
    ax.set_title(case.title, loc="left")
    fig.tight_layout()
    style.save(fig, path)
    plt.close(fig)
    return path


def _curves(nports: int) -> list[tuple[int, int]]:
    """First column of S: reflection at port 1 and transmission from it. For a
    reciprocal structure this is the information a README row has room for."""
    return [(j, 0) for j in range(min(nports, 4))]


def sparams(case: Case, path: Path) -> Path:
    style.setup()
    import matplotlib.pyplot as plt

    f_min, f_max = case.band
    fig, (mag, pha) = plt.subplots(2, 1, figsize=style.FIG_SIZE, sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    sources = [(m.label, m.network(), "-") for m in case.measurements()]
    sources += [(name, _load(ts), "--") for name, _, ts in case.results()]
    for label, net, ls in sources:
        keep = (net.f >= f_min) & (net.f <= f_max)
        f = net.f[keep] / 1e9
        for k, (i, j) in enumerate(_curves(net.nports)):
            s = net.s[keep, i, j]
            color = style.SERIES[k]
            name = f"S{i + 1}{j + 1}" if ls == "-" else f"S{i + 1}{j + 1} {label}"
            mag.plot(f, 20 * np.log10(np.abs(s)), ls=ls, color=color, label=name)
            pha.plot(f, np.degrees(np.angle(s)), ls=ls, color=color)
    for ax in (mag, pha):
        ax.grid(True)
        ax.set_xlim(f_min / 1e9, f_max / 1e9)
    mag.set_ylabel("|S| [dB]")
    pha.set_ylabel("phase [deg]")
    pha.set_ylim(-180, 180)
    pha.set_yticks([-180, -90, 0, 90, 180])
    pha.set_xlabel("f [GHz]")
    mag.legend(loc="best", fontsize=style.FONT_SIZE - 1)
    mag.set_title("measured" if len(sources) == len(case.measurements())
                  else "measured (solid) and solver results (dashed)",
                  loc="left")
    fig.tight_layout()
    style.save(fig, path)
    plt.close(fig)
    return path


def _load(path: Path):
    import skrf
    return skrf.Network(str(path))
