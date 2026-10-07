"""Scalar quantities derived from S-parameters, by the names the case files use.

A case can ask for its measurement to be shown as L and Q instead of S-parameters
(case.yaml: plot); the expressions are evaluated here.
"""
from __future__ import annotations

import re

import numpy as np

UNITS = {"db": "dB", "deg": "deg", "L": "nH", "R": "Ohm", "Q": "1"}
_EXPR = re.compile(r"^(db|deg)\(S([1-9])([1-9])\)$|^(L|Q|R)\((Y11|Y22|Z11|Z22|Zdiff|Zser)\)$")


def unit(expr: str) -> str:
    m = _EXPR.match(expr)
    return UNITS[m.group(1) or m.group(4)]


def impedance(net, which: str) -> np.ndarray:
    z, y = net.z, net.y
    return {
        "Y11": 1 / y[:, 0, 0],
        "Y22": 1 / y[:, 1, 1],
        "Z11": z[:, 0, 0],
        "Z22": z[:, 1, 1],
        "Zdiff": z[:, 0, 0] + z[:, 1, 1] - z[:, 0, 1] - z[:, 1, 0],
        "Zser": -1 / y[:, 0, 1],
    }[which]


def evaluate(net, expr: str) -> np.ndarray:
    m = _EXPR.match(expr)
    if m is None:
        raise ValueError(f"unknown quantity {expr!r}")
    if m.group(1):
        s = net.s[:, int(m.group(2)) - 1, int(m.group(3)) - 1]
        return 20 * np.log10(np.abs(s)) if m.group(1) == "db" else np.degrees(np.angle(s))
    x = impedance(net, m.group(5))
    return {"L": x.imag / (2 * np.pi * net.f) * 1e9, "R": x.real,
            "Q": x.imag / x.real}[m.group(4)]
