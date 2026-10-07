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
    # Metals bottom-up so an underpass stays visible; vias on top, they are small.
    order = sorted(layers, key=lambda n: (stack.conductor(n)["kind"] == "via", stack.z(n)[0]))
    metals = [n for n in order if stack.conductor(n)["kind"] == "metal"]

    fig, ax = plt.subplots(figsize=style.FIG_SIZE)
    for name in order:
        if stack.conductor(name)["kind"] == "via":
            face, alpha = style.INK, 0.9
        else:
            face, alpha = style.SERIES[metals.index(name) % len(style.SERIES)], 0.85
        ax.add_collection(PolyCollection(layers[name], facecolors=face, edgecolors=face,
                                         linewidths=0.3, alpha=alpha, label=name))
    for poly in case.outline() or []:
        ax.add_collection(PolyCollection([poly], facecolors="none", edgecolors=style.INK,
                                         linewidths=0.8, label="board outline"))
    for p in case.ports():
        if p.deembed_um:
            plane = _shifted(p, layers[p.layer])
            ax.plot(*zip(*plane), color=style.INK, lw=1, ls="--", zorder=5)
        (ax_, ay), (bx, by) = p.segment
        ax.plot([ax_, bx], [ay, by], color=style.INK, lw=3, solid_capstyle="butt", zorder=5)
        ax.plot(*p.midpoint, marker="o", ms=4, color=style.INK, zorder=5)
        ax.annotate(p.name, p.midpoint, textcoords="offset points", xytext=(5, -10),
                    color=style.INK, fontsize=style.FONT_SIZE)
    ax.autoscale_view()
    ax.margins(0.06)
    # A long line drawn to scale is a hairline. Beyond MAX_ASPECT the y axis is
    # stretched, and the label says by how much.
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    ratio = max((x1 - x0) / (y1 - y0), (y1 - y0) / (x1 - x0))
    stretch = ratio / MAX_ASPECT if ratio > MAX_ASPECT else 1.0
    wide = (x1 - x0) >= (y1 - y0)
    ax.set_aspect(stretch if wide else 1 / stretch)
    ax.set_xlabel("x [um]" + ("" if wide or stretch == 1 else f", stretched x{stretch:.0f}"))
    ax.set_ylabel("y [um]" + ("" if not wide or stretch == 1 else f", stretched x{stretch:.0f}"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=style.FONT_SIZE - 1)
    fig.tight_layout()
    style.save(fig, path, tight=True)
    plt.close(fig)
    return path


MAX_ASPECT = 3.0       # longest-to-shortest side of a layout plot


def _shifted(port, polygons) -> list[tuple[float, float]]:
    """The reference plane of a de-embedded line port: the segment moved by
    deembed_um along its normal, towards the metal."""
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    (ax, ay), (bx, by) = port.segment
    length = np.hypot(bx - ax, by - ay)
    nx, ny = (ay - by) / length, (bx - ax) / length
    mx, my = port.midpoint
    metal = unary_union([Polygon(q) for q in polygons])
    if not metal.contains(Point(mx + nx * 1e-3 * length, my + ny * 1e-3 * length)):
        nx, ny = -nx, -ny
    d = port.deembed_um
    return [(ax + nx * d, ay + ny * d), (bx + nx * d, by + ny * d)]


def _curves(nports: int) -> list[tuple[int, int]]:
    """First column of S: reflection at port 1 and transmission from it. For a
    reciprocal structure this is the information a README row has room for."""
    return [(j, 0) for j in range(min(nports, 4))]


def sparams(case: Case, path: Path) -> Path:
    if case.raw.get("plot") or any(m.quantity == "curves" for m in case.measurements()):
        return curves(case, path)
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


def curves(case: Case, path: Path) -> Path:
    """Measured curves (L, Q, |S| ...) in up to two panels, one per unit. Solver
    results are evaluated to the same quantities and drawn dashed."""
    from . import quantities
    style.setup()
    import matplotlib.pyplot as plt

    f_min, f_max = case.band
    series = []
    for m in case.measurements():
        if m.quantity == "curves":
            series += [(m, name, expr, f, v) for name, (expr, f, v) in m.curves().items()]
        else:
            net = m.network()
            net = net[(net.f >= f_min) & (net.f <= f_max) & (net.f > 0)]
            series += [(m, m.label, expr, net.f, quantities.evaluate(net, expr))
                       for expr in case.raw.get("plot", [])]
    units = list(dict.fromkeys(quantities.unit(expr) for _, _, expr, _, _ in series))[:2]
    fig, axes = plt.subplots(len(units), 1, figsize=style.FIG_SIZE, sharex=True, squeeze=False)
    axes = {u: ax for u, ax in zip(units, axes[:, 0])}
    results = [(name, _load(ts)) for name, _, ts in case.results()]
    for k, (m, name, expr, f, v) in enumerate(series):
        ax = axes.get(quantities.unit(expr))
        if ax is None:
            continue
        # colour follows the entity: the measurement when there are several, else the series
        labels = [x.label for x in case.measurements()]
        k = labels.index(m.label) if len(labels) > 1 else k
        color = style.SERIES[k % len(style.SERIES)]
        marker = "." if m.origin == "digitized" else None
        label = expr if len(case.measurements()) == 1 else f"{expr} {m.label}"
        ax.plot(f / 1e9, v, color=color, marker=marker, ms=3, label=label)
        for rname, net in results:
            keep = (net.f >= f_min) & (net.f <= f_max) & (net.f > 0)
            ax.plot(net.f[keep] / 1e9, quantities.evaluate(net[keep], expr), ls="--",
                    color=color, label=f"{expr} {rname}")
    for u, ax in axes.items():
        # L and Q diverge at self-resonance; scale to the bulk of the data
        vals = np.concatenate([v for _, _, e, _, v in series if quantities.unit(e) == u])
        lo, hi = np.percentile(vals, [3, 97])
        pad = 0.1 * (hi - lo or 1.0)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(True)
        ax.set_xlim(f_min / 1e9, f_max / 1e9)
        ax.set_ylabel(u if u != "1" else "")
        ax.legend(loc="best", fontsize=style.FONT_SIZE - 1)
    list(axes.values())[-1].set_xlabel("f [GHz]")
    figs = sorted({m.meta["digitized"]["figure"] for m, *_ in series if m.origin == "digitized"})
    title = "measured" + (f", digitized from {', '.join(figs)}" if figs else "")
    if results:
        title += "; solver results dashed"
    list(axes.values())[0].set_title(title, loc="left")
    fig.tight_layout()
    style.save(fig, path)
    plt.close(fig)
    return path


def _load(path: Path):
    import skrf
    return skrf.Network(str(path))
