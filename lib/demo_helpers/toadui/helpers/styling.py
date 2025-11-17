#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from types import SimpleNamespace

from .types import SelfType


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class UIStyle(SimpleNamespace):
    def copy(self):
        return UIStyle(**self.__dict__)

    def update(self, **kwargs) -> SelfType:
        """Function used to update multiple style attributes at once"""
        for key, val in kwargs.items():
            assert key in self.__dict__, f"Invalid style attribute: {key}"
            self.__dict__[key] = val
        return self

    def __call__(self, **kwargs) -> SelfType:
        """Update multiple style attributes at once. See .update(...) function"""
        return self.update(**kwargs)


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def get_background_thickness(foreground_thickness: int) -> int:
    """
    Helper used to choose a 'background' thickness for use in plotting
    lines or text with opencv. The goal being to create an outline effect,
    ideally about 1px around the drawn item. The rules for doing this seem
    to vary slightly with the original thickness
    """
    return foreground_thickness + 1 + (foreground_thickness % 2) if foreground_thickness > 1 else 2
