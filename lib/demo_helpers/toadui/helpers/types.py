#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from typing import NamedTuple, TypeAlias, Callable, Any

try:
    from typing import Self
except (ImportError, ModuleNotFoundError):
    Self: TypeAlias = None


# ---------------------------------------------------------------------------------------------------------------------
# %% Tuples


class XYPairPX(NamedTuple):
    x: int
    y: int


class XYPairNorm(NamedTuple):
    x: float
    y: float


class HWPairPX(NamedTuple):
    h: int
    w: int


class LMRData(NamedTuple):
    left: Any
    middle: Any
    right: Any


class IsLMR(NamedTuple):
    left: bool
    middle: bool
    right: bool


# ---------------------------------------------------------------------------------------------------------------------
# %% Aliases

XYPX: TypeAlias = tuple[int, int]
XYNORM: TypeAlias = tuple[float, float]

XY1XY2PX: TypeAlias = tuple[XYPX, XYPX]
XY1XY2NORM: TypeAlias = tuple[XYNORM, XYNORM]

COLORU8: TypeAlias = tuple[int, int, int]
COLORNORM: TypeAlias = tuple[float, float, float]

HWPX: TypeAlias = tuple[int, int]
IMGSHAPE_HW: TypeAlias = tuple[int, int] | tuple[int, int, int]

SelfType: TypeAlias = Self

EmptyCallback: TypeAlias = Callable[[], None]
