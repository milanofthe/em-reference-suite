"""Read measured curves off vector figures in a PDF.

Most journal PDFs keep their plots as vector paths. Reading those paths is exact up
to the line width, where reading pixels is not. The procedure:

1. frames(): the plot rectangles on a page, so an importer can name the one it wants.
2. axes(): calibrate both axes from the numeric tick labels around a frame. All
   labels on a side enter a least-squares fit, linear or logarithmic, whichever
   leaves the smaller residual; the residual is reported as the reading accuracy.
3. paths(): every stroked path inside the frame, grouped by stroke colour and
   mapped to data coordinates.

Coordinates are pdfplumber's: x to the right, top downwards, in points.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Axis:
    scale: str            # "linear" | "log"
    a: float              # value = a * pos + b, on log10(value) for "log"
    b: float
    residual: float       # rms misfit of the tick labels, in axis units (decades for log)

    @property
    def resolution(self) -> float:
        """Reading uncertainty from placing labels and ticks to about half a point,
        in axis units (decades for a log axis)."""
        return abs(self.a) * 0.5 + self.residual

    def value(self, pos: np.ndarray) -> np.ndarray:
        v = self.a * np.asarray(pos) + self.b
        return 10 ** v if self.scale == "log" else v


def _page(pdf_path, page_no: int):
    import pdfplumber
    return pdfplumber.open(str(pdf_path)).pages[page_no - 1]


def frames(pdf_path, page_no: int, min_size: float = 40.0) -> list[tuple[float, float, float, float]]:
    """(x0, top, x1, bottom) of every stroked rectangle that can hold a plot."""
    page = _page(pdf_path, page_no)
    out = [(r["x0"], r["top"], r["x1"], r["bottom"]) for r in page.rects
           if r["width"] >= min_size and r["height"] >= min_size
           and r["width"] < 0.9 * page.width]
    return sorted(set((round(a, 1), round(b, 1), round(c, 1), round(d, 1)) for a, b, c, d in out))


def _number(text: str) -> float | None:
    t = text.replace("−", "-").replace("–", "-")
    try:
        return float(t)
    except ValueError:
        return None


def _fit(pos: np.ndarray, val: np.ndarray) -> Axis:
    """Best of a linear and a log fit. Labels that do not sit on the fitted scale
    (a stray number from a caption or a neighbouring plot) are dropped one at a
    time while at least three remain."""
    best = None
    for scale in ("linear", "log"):
        if scale == "log" and np.any(val <= 0):
            continue
        v = np.log10(val) if scale == "log" else val
        p_, v_ = pos.copy(), v.copy()
        while True:
            a, b = np.polyfit(p_, v_, 1)
            err = np.abs(a * p_ + b - v_)
            if len(p_) <= 3 or err.max() < 1e-3 * (np.ptp(v_) or 1.0):
                break
            keep = np.arange(len(p_)) != err.argmax()
            p_, v_ = p_[keep], v_[keep]
        res = float(np.sqrt(np.mean((a * p_ + b - v_) ** 2)))
        span = np.ptp(v) or 1.0
        if best is None or res / span < best.residual / (best_span or 1.0):
            best, best_span = Axis(scale, float(a), float(b), res), span
    return best


def axes(pdf_path, page_no: int, frame, margin: float = 30.0) -> tuple[Axis, Axis]:
    """Calibrate x from the numbers below the frame and y from those to its left."""
    page = _page(pdf_path, page_no)
    x0, top, x1, bottom = frame
    xs, ys = [], []
    # Rotated axis titles ("|S11|" set vertically) would merge with tick labels.
    # pdfplumber's "upright" flag is unreliable; the text matrix is not.
    upright = page.filter(lambda o: o.get("object_type") != "char"
                          or abs(o["matrix"][1]) < abs(o["matrix"][0]))
    for w in upright.extract_words():
        v = _number(w["text"])
        if v is None:
            continue
        cx, cy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
        if x0 - 5 <= cx <= x1 + 5 and bottom < w["top"] < bottom + margin:
            xs.append((cx, v))
        elif top - 5 <= cy <= bottom + 5 and x0 - margin < w["x1"] < x0:
            ys.append((cy, v))
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError(f"too few tick labels: {len(xs)} on x, {len(ys)} on y")
    (px, vx), (py, vy) = (np.array(xs).T, np.array(ys).T)
    return _fit(px, vx), _fit(py, vy)


def paths(pdf_path, page_no: int, frame, x: Axis, y: Axis,
          min_points: int = 4) -> dict[str, list[np.ndarray]]:
    """Stroked paths inside the frame, by stroke colour, as arrays of (x, y) data."""
    page = _page(pdf_path, page_no)
    x0, top, x1, bottom = frame
    out: dict[str, list[np.ndarray]] = {}
    for c in page.curves + page.lines:
        if not c.get("stroke"):
            continue
        pts = np.array(c.get("pts") or [(c["x0"], c["top"]), (c["x1"], c["bottom"])], float)
        inside = ((pts[:, 0] >= x0 - 0.5) & (pts[:, 0] <= x1 + 0.5)
                  & (pts[:, 1] >= top - 0.5) & (pts[:, 1] <= bottom + 0.5))
        if inside.sum() < min_points:
            continue
        pts = pts[inside]
        out.setdefault(str(c.get("stroking_color")), []).append(
            np.column_stack([x.value(pts[:, 0]), y.value(pts[:, 1])]))
    return out
