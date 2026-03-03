# -*- coding: utf-8 -*-
"""Coordinate conversion utilities. Convention: a1 = (0,0), i10 = (8,9)."""

from __future__ import annotations
from typing import Tuple

from .config import BOARD_W, BOARD_H

FILES = "abcdefghi"  # 9 files
assert len(FILES) == BOARD_W


def to_alg(x: int, y: int) -> str:
    """(x,y) -> 'a1'..'i10'"""
    if not (0 <= x < BOARD_W and 0 <= y < BOARD_H):
        raise ValueError(f"out of bounds: {(x,y)}")
    return f"{FILES[x]}{y+1}"


def from_alg(s: str) -> Tuple[int, int]:
    """'a1'..'i10' -> (x,y)"""
    s = s.strip().lower()
    if len(s) < 2:
        raise ValueError(f"bad coord: {s}")
    file_ch = s[0]
    if file_ch not in FILES:
        raise ValueError(f"bad file: {file_ch}")
    try:
        rank = int(s[1:])
    except ValueError as e:
        raise ValueError(f"bad rank in {s}") from e
    if not (1 <= rank <= BOARD_H):
        raise ValueError(f"rank out of range: {rank}")
    x = FILES.index(file_ch)
    y = rank - 1
    return x, y
