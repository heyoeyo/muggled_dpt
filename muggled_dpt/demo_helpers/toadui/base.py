#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from dataclasses import dataclass

import cv2
import numpy as np

# For type hints
from typing import Protocol, Iterator
from numpy import ndarray
from .helpers.types import HWPX, SelfType, XYPairNorm, XYPairPX, HWPairPX
from .helpers.ocv_types import OCVEvent


# ---------------------------------------------------------------------------------------------------------------------
# %% Data types


@dataclass(frozen=True)
class CBEventFlags:

    ctrl_key: bool
    shift_key: bool
    alt_key: bool

    @classmethod
    def create(cls, cv2_flags: int):
        # -> Flags are provided as single integer, where bit positions encode states
        #    so multiple states can be active at the same time
        # -> Can determine which states/flags are active through bitwise filtering
        return cls(
            bool(cv2_flags & cv2.EVENT_FLAG_CTRLKEY),
            bool(cv2_flags & cv2.EVENT_FLAG_SHIFTKEY),
            bool(cv2_flags & cv2.EVENT_FLAG_ALTKEY),
        )

    @classmethod
    def default(cls):
        return cls(False, False, False)


@dataclass(frozen=True)
class CBEventXY:

    xy_px: XYPairPX
    xy_norm: XYPairNorm
    hw_px: HWPairPX
    is_in_region: bool

    @classmethod
    def default(cls):
        return cls(XYPairPX(-1, 1), XYPairNorm(-1, -1), HWPairPX(-1, -1), False)


@dataclass(frozen=False)
class CBRegion:

    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    w: int = 1
    h: int = 1

    def resize(self, x1: int, y1: int, x2: int, y2: int) -> SelfType:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.w = max(1, self.x2 - self.x1)
        self.h = max(1, self.y2 - self.y1)
        return self

    def make_cbeventxy(self, global_x_px: int, global_y_px: int) -> CBEventXY:
        """Convenience function which does multiple internal conversions (all together for efficiency)"""

        x_px, y_px = global_x_px - self.x1, global_y_px - self.y1
        x_norm = (global_x_px - self.x1) / self.w
        y_norm = (global_y_px - self.y1) / self.h
        is_in_region = (self.x1 <= global_x_px < self.x2) and (self.y1 <= global_y_px < self.y2)

        return CBEventXY(XYPairPX(x_px, y_px), XYPairNorm(x_norm, y_norm), HWPairPX(self.h, self.w), is_in_region)


@dataclass(frozen=False)
class CBState:
    """Container for basic callback state"""

    disabled: bool = False
    hovered: bool = False
    left_pressed: bool = False
    middle_pressed: bool = False
    right_pressed: bool = False
    event_xy: CBEventXY = CBEventXY.default()


@dataclass(frozen=True)
class CBRenderSizing:

    min_h: int = 32
    min_w: int = 32
    is_flexible_h: bool = False
    is_flexible_w: bool = False

    def copy(self):
        """Create new copy of the current render sizing object"""
        return CBRenderSizing(self.min_h, self.min_w, self.is_flexible_h, self.is_flexible_w)


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class CBChild(Protocol):
    """Protocol used to help type hint BaseCallback methods"""

    def _cb_iter(self, global_x_px: int, global_y_px: int) -> None: ...


class BaseCallback(CBChild):
    """
    Main class of all UI elements. Anything that intended to
    respond to mouse events should inherit from this class!
    """

    def __init__(self, min_h: int, min_w: int, is_flexible_h=False, is_flexible_w=False):

        # Storage for rendering info
        assert min_h > 0, "Must have non-zero minimum height!"
        assert min_w > 0, "Must have non-zero minimum width!"
        self._cb_rdr = CBRenderSizing(min_h, min_w, is_flexible_h, is_flexible_w)

        # Storage for placement and state of callback
        self._cb_region = CBRegion()
        self._cb_state = CBState()

        # Storage for all child callback items
        self._cb_parent_list: list[BaseCallback] = []
        self._cb_child_list: list[BaseCallback] = []

        # Storage for name that can be printed when debugging
        self._debug_name: str | None = None

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        child_names = [str(child) for child in self]
        return f"{cls_name} [{', '.join(child_names)}]" if len(child_names) > 0 else cls_name

    # .................................................................................................................

    def _set_debug_name(self, name: str) -> SelfType:
        """
        Helper used for debugging. Assigns a 'debug name' used by
        the ._print_debug(...) function
        """
        self._debug_name = name
        return self

    def _print_debug(self, *values, sep=" ", flush=False):
        """
        Helper used for debugging. Acts like a regular print statement,
        but will prefix a 'debug name' to each printed line.
        """
        debug_name = self._debug_name if self._debug_name is not None else self.__class__.__name__[0:4]
        print(f"{debug_name} |", *values, sep=sep, flush=flush)
        return None

    # .................................................................................................................

    def enable(self, enable_callback=True) -> SelfType:
        self._cb_state.disabled = not enable_callback
        return self

    def is_hovered(self) -> bool:
        return self._cb_state.hovered

    def get_event_xy(self) -> CBEventXY:
        return self._cb_state.event_xy

    def get_min_hw(self) -> HWPairPX:
        return HWPairPX(self._cb_rdr.min_h, self._cb_rdr.min_w)

    def render(self, h: int | None = None, w: int | None = None) -> ndarray:
        rend_h, rend_w = self._get_render_hw(h, w)
        frame = self._render_up_to_size(rend_h, rend_w)

        # Sanity check that render target did what it was told to do...
        if len(self._cb_parent_list) > 0:
            is_correct_size = (frame.shape[0] == h) and (frame.shape[1] == w)
            assert is_correct_size, f"Bad render size: {tuple(frame.shape[0:2])} vs {(h, w)} ({self})"
        else:
            # Set the correct callback region sizing for the caller element
            # (as called likely has no layout element to update it's sizing!)
            img_h, img_w = frame.shape[0:2]
            if img_h != self._cb_region.h or img_w != self._cb_region.w:
                self._update_cb_region(0, 0, img_w, img_h)
            pass
        pass

        return frame

    # .................................................................................................................

    def _append_cb_children(self, *child_items: CBChild | ndarray | None) -> SelfType:
        """
        Used to 'store' other callback elements. Expected to be used
        when building layouts, consisting of many (nested) elements.
        When events are triggered on the parent element, they will
        be passed to all appended child elements.
        Items given as 'None' will be skipped, this can be used to
        conditionally disable entries.
        """

        for child in child_items:

            # Skip None entries for conditional disabling
            if child is None:
                continue

            # Assume numpy arrays are images and should be treated as such
            if isinstance(child, np.ndarray):
                child = _RawImageWrapperElement(child)

            # Sanity check. Make sure we're dealing with a callback item before storing
            assert isinstance(child, BaseCallback), f"Children must inherit from: BaseCallback, got: {type(child)}"
            self._cb_child_list.append(child)
            self._register_as_parent_of(child)

        return self

    def _register_as_parent_of(self, *child_items: CBChild) -> SelfType:
        """
        Used to record 'self' as a parent to the provided child element(s).
        The primary purpose for this (at least so far) is to indicate to
        child elements that they are 'nested' within other elements. This
        is important, for example, for layout elements so that they do
        not assume they are the 'outer-most' element when rendering.
        """
        for child in child_items:
            assert self not in child._cb_parent_list, f"Error - {self} is already a parent of {child}"
            child._cb_parent_list.append(self)
        return self

    def _clear_cb_children(self) -> SelfType:
        """
        Function used to clear existing child callbacks.
        This has the potential to cause major problems with proper
        handling of callbacks! Only use if absolutely neccessary.
        """
        self._cb_child_list: list[BaseCallback] = []
        return self

    def _clear_cb_parent(self, parent=None) -> SelfType:
        """
        Function used to clear an existing parent listing.
        This should normally not be called, but may be useful
        in cases where a different 'parent' element is being
        substituted to take over child elements.

        If 'None' is provided, then all parents will be
        cleared (this is not recommended!), otherwise only
        the provided parent will be cleared, and only if
        it is in the parent listing

        """
        if parent is None:
            self._cb_parent_list: list[BaseCallback] = []
        elif parent in self._cb_parent_list:
            parent_idx = self._cb_parent_list.index(parent)
            self._cb_parent_list.pop(parent_idx)
        return self

    def __len__(self) -> int:
        return len(self._cb_child_list)

    def __iter__(self):
        return iter(self._cb_child_list)

    def __getitem__(self, index) -> CBChild:
        return self._cb_child_list[index]

    # .................................................................................................................

    def _on_move(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def _on_drag(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def _on_left_click(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def _on_left_down(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def _on_left_up(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def _on_left_double(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def _on_right_click(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def _on_right_down(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def _on_right_up(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def _on_right_double(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def _on_middle_click(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def _on_middle_down(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def _on_middle_up(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def _on_middle_double(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    def _on_mouse_wheel(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        return

    # .................................................................................................................

    def _on_opencv_event(self, event: OCVEvent, x: int, y: int, flags: int, params: None) -> None:

        # Disable callback handling if needed
        if self._cb_state.disabled:
            return

        # Precompute flag states for easier handling
        cbflags = CBEventFlags.create(flags)

        # Big ugly if-else to handle all possible events
        if event == cv2.EVENT_MOUSEMOVE:
            for cbitem, cbxy in self._cb_iter(x, y):
                cbitem._cb_state.hovered = cbxy.is_in_region
                cbitem._cb_state.event_xy = cbxy
                if cbitem._cb_state.left_pressed:
                    cbitem._on_drag(cbxy, cbflags)
                cbitem._on_move(cbxy, cbflags)

        elif event == cv2.EVENT_LBUTTONDOWN:
            for cbitem, cbxy in self._cb_iter(x, y):
                cbitem._cb_state.left_pressed = cbxy.is_in_region
                cbitem._on_left_down(cbxy, cbflags)

        elif event == cv2.EVENT_RBUTTONDOWN:
            for cbitem, cbxy in self._cb_iter(x, y):
                cbitem._cb_state.right_pressed = cbxy.is_in_region
                cbitem._on_right_down(cbxy, cbflags)

        elif event == cv2.EVENT_MBUTTONDOWN:
            for cbitem, cbxy in self._cb_iter(x, y):
                cbitem._cb_state.middle_pressed = cbxy.is_in_region
                cbitem._on_middle_down(cbxy, cbflags)

        elif event == cv2.EVENT_LBUTTONUP:
            for cbitem, cbxy in self._cb_iter(x, y):
                if cbitem._cb_state.left_pressed:
                    cbitem._cb_state.left_pressed = False
                    if cbxy.is_in_region:
                        cbitem._on_left_click(cbxy, cbflags)
                cbitem._on_left_up(cbxy, cbflags)

        elif event == cv2.EVENT_RBUTTONUP:
            for cbitem, cbxy in self._cb_iter(x, y):
                if cbitem._cb_state.right_pressed:
                    cbitem._cb_state.right_pressed = False
                    if cbxy.is_in_region:
                        cbitem._on_right_click(cbxy, cbflags)
                cbitem._on_right_up(cbxy, cbflags)

        elif event == cv2.EVENT_MBUTTONUP:
            for cbitem, cbxy in self._cb_iter(x, y):
                if cbitem._cb_state.middle_pressed:
                    cbitem._cb_state.middle_pressed = False
                    if cbxy.is_in_region:
                        cbitem._on_middle_click(cbxy, cbflags)
                cbitem._on_middle_up(cbxy, cbflags)

        if event == cv2.EVENT_LBUTTONDBLCLK:
            for cbitem, cbxy in self._cb_iter(x, y):
                cbitem._cb_state.left_pressed = False
                cbitem._on_left_double(cbxy, cbflags)

        elif event == cv2.EVENT_RBUTTONDBLCLK:
            for cbitem, cbxy in self._cb_iter(x, y):
                cbitem._cb_state.right_pressed = False
                cbitem._on_right_double(cbxy, cbflags)

        elif event == cv2.EVENT_MBUTTONDBLCLK:
            for cbitem, cbxy in self._cb_iter(x, y):
                cbitem._cb_state.middle_pressed = False
                cbitem._on_middle_double(cbxy, cbflags)

        elif event == cv2.EVENT_MOUSEWHEEL:
            for cbitem, cbxy in self._cb_iter(x, y):
                cbitem._on_mouse_wheel(cbxy, cbflags)

        return

    def _cb_iter(self, global_x_px: int, global_y_px: int) -> Iterator[tuple[SelfType | CBChild, CBEventXY]]:
        """Helper used to run callbacks on all self + children"""

        # Return our own event data
        cbxy = self._cb_region.make_cbeventxy(global_x_px, global_y_px)
        if not self._cb_state.disabled:
            yield self, cbxy

        # Recursively call iterator on all children and children-of-children etc. to call all nested callbacks
        for child in self._cb_child_list:
            if not child._cb_state.disabled:
                yield from child._cb_iter(global_x_px, global_y_px)

        return

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:
        raise NotImplementedError(f"Must implement '_render_up_to_size' function ({self})")

    def _get_render_hw(self, h: int | None = None, w: int | None = None) -> HWPX:

        if h is not None and w is not None:
            h = max(h, self._cb_rdr.min_h)
            w = max(w, self._cb_rdr.min_w)

        elif h is None and w is None:
            h, w = self._get_height_and_width_without_hint()
        elif h is None:
            h = self._get_height_given_width(w)
        elif w is None:
            w = self._get_width_given_height(h)

        return (h, w)

    # .................................................................................................................

    def _get_width_given_height(self, h: int) -> int:
        """Function used to communicate how wide an element will be, if asked to render to a given height"""
        return self._cb_rdr.min_w

    def _get_height_given_width(self, w: int) -> int:
        """Function used to communicate how tall an element will be, if asked to render to a given width"""
        return self._cb_rdr.min_h

    def _get_height_and_width_without_hint(self) -> HWPX:
        """Function used to communicate how wide & tall an element will be, if no size is specified"""
        return self._cb_rdr.min_h, self._cb_rdr.min_w

    def _get_dynamic_aspect_ratio(self) -> float | None:
        """
        Function used to report the aspect ratio of an element's
        non-fixed sized components, if any, for rendering purposes.
        If an element does not have a target aspect ratio, for example a button,
        then it should return None. By comparison, an image may render at a
        fixed aspect ratio, and should report that number instead.

        Other elements may include a fixed sized element, along with a
        resizing element, in this case, only the aspect ratio of the
        dynamic element component should be reported.
        """
        return None

    def _update_cb_region(self, x1, y1, x2, y2) -> SelfType:
        """
        Function used to resize callback region (used to look for mouse events).
        This is generally called by layout elements, which need to tell child elements
        about changes to their positioning or size.

        Note that position/sizing is often not known until *after* an element has rendered.
        This can lead to errors in parent-child region sizing.
        For example consider a sequence involving a top-most element 'T' which contains
        a child element 'A' which itself contains a child 'B':
            Assume Item A has initial region: (x1=0, y1=0, x2=400, y2=500)
            1. Item T (top-most) asks item A (child of T) to render
            2. Item A renders, during which item B (child of A) is asked to render.
               Item B ends up having a height of 100 and width of 200
            3. Item A asks item B to update region to (x1,y1,x1+200,y1+100)
               where x1,y1 come from A's region, so B has region: (0,0,200,100)
            4. Item T asks item A to update region to (100,100,900,800)

        This last update, which offsets and scales the region of item A (e.g. x1=100, y1=100),
        now means that item B has the wrong region because it was offset using the original
        x1/y1 of item A which is now changed by the last update. This may be fixed on the
        next render update, where item A will now pass the updated x1/y1 offsets to item B,
        but if A is caching it's render results, this may not happen right away!

        This function therefore can be overriden to resolve these situations. In this
        case, item A would override it's resizing to also update item B, so that any
        time the A region changes, it also updates the B region. This should only
        be needed when creating elements that have tight parent-child sizing.
        """
        self._cb_region.resize(x1, y1, x2, y2)

    # .................................................................................................................


class BaseOverlay(BaseCallback):
    """
    Callback element which is meant to sit 'on top' of another item.
    Rendering is handled through a '_render_overlay' callback, which
    is given the underlying (rendered) item result as an input.

    These elements are intended to be used for rendering dynamic indicators.
    If many overlays are being used together, consider using an OverlayStack
    layout element!
    """

    # .................................................................................................................

    def __init__(self, base_item: BaseCallback | None, suppress_callbacks_to_base: bool = False):
        """
        The base_item is meant to be a display item that the overlay 'draws on top of'.

        The input flag suppress_callbacks_to_base will prevent callbacks from being passed
        to the base item, so that only the overlay is interactive.

        A base_item of 'None' can be provided, but is only intended for a special use case!
        Specifically, if this overlay is part of a stack of overlays on a single base item,
        then a 'None' value can be provided. The assumption being that, later, the parent
        stack element will be responsible for handling the rendering and interaction of
        all overlays elements (this is mainly for the sake of optimization).
        """

        super().__init__(1, 1)
        self._enable_overlay_render = True
        self._base_item = base_item
        if base_item is not None:
            if not suppress_callbacks_to_base:
                self._append_cb_children(self._base_item)
            self._cb_rdr = base_item._cb_rdr.copy()
        pass

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name}" if self._base_item is None else f"{cls_name}: {self._base_item}"

    def _render_overlay(self, frame: ndarray) -> ndarray:
        raise NotImplementedError(f"Must implement '_render_overlay' function ({self})")

    # .................................................................................................................

    def get_base_item(self) -> BaseCallback | None:
        """
        Function used to get access to the underlying base element, if present.
        Note that the base item may be set to 'None', usually when used in
        an overlay stack (e.g. more than 1 overlay on top of a single item)!
        """
        return self._base_item

    # .................................................................................................................

    def enable_render(self, enable_render: bool | None = None) -> SelfType:
        self._enable_overlay_render = (not self._enable_overlay_render) if enable_render is None else enable_render
        return self

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:
        """
        Special overlay render function, which 'asks' the base item to render
        first, then renders an overlay on top of this.
        Generally, this should NOT be overridden by classes inheriting from this classes!
        -> Instead override the _render_overlay(...) function
        """

        # Sanity check. If no base item was given, we don't expect to call this function
        # -> Instead some parent item should manage rendering and call _render_overlay instead!
        assert self._base_item is not None, f"No base item provided for rendering overlay! ({self})"

        # Set up starting stack point, used to keep track of child callback regions
        x1 = self._cb_region.x1
        y1 = self._cb_region.y1

        # Have base item provide the base frame rendering and overlays handle drawing over-top
        base_frame = self._base_item._render_up_to_size(h, w).copy()
        base_h, base_w = base_frame.shape[0:2]

        x2, y2 = x1 + base_w, y1 + base_h
        self._base_item._update_cb_region(x1, y1, x2, y2)
        self._update_cb_region(x1, y1, x2, y2)

        return self._render_overlay(base_frame) if self._enable_overlay_render else base_frame

    # .................................................................................................................

    def _get_dynamic_aspect_ratio(self):
        return self._base_item._get_dynamic_aspect_ratio()

    def _get_height_and_width_without_hint(self) -> HWPX:
        return self._base_item._get_height_and_width_without_hint()

    def _get_height_given_width(self, w: int) -> int:
        return self._base_item._get_height_given_width(w)

    def _get_width_given_height(self, h: int) -> int:
        return self._base_item._get_width_given_height(h)

    # .................................................................................................................


class CachedBgFgElement(BaseCallback):
    """
    Helper wrapper around the base callback class,
    which includes a simple background-foreground-postprocess
    template for rendering. Elements inheriting from this
    class are expected to override:
        _rerender_bg, _rerender_fg and (optionally) _post_rerender

    Note that the _post_rerender function is meant to be a catch-all
    pass which allows for per-frame dynamic changes, if needed. For
    example, it can be used to render hover effects.
    """

    # .................................................................................................................

    def __init__(self, min_h: int, min_w: int, is_flexible_h: bool = False, is_flexible_w: bool = False):
        super().__init__(min_h, min_w, is_flexible_h, is_flexible_w)
        self._cache_h = -1
        self._cache_w = -1
        self._cached_bg_img = np.zeros((1, 1, 3), dtype=np.uint8)
        self._cached_fg_img = np.zeros((1, 1, 3), dtype=np.uint8)
        self._needs_fg_repaint = True

    # .................................................................................................................

    def request_full_repaint(self, need_repaint=True) -> SelfType:
        """Force a re-paint of both the background & foreground images"""
        if need_repaint:
            self._cache_h = -1
            self._cache_w = -1
            self._cached_bg_img = np.zeros((1, 1, 3), dtype=np.uint8)
        return self

    def request_fg_repaint(self, need_repaint=True) -> SelfType:
        """Function used to re-paint the cached foreground image on the next render update"""
        self._needs_fg_repaint = need_repaint
        return self

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Re-render background on size change
        if self._cache_h != h or self._cache_w != w:
            self._cache_h = h
            self._cache_w = w
            self._cached_bg_img = self._rerender_bg(h, w)
            self._needs_fg_repaint = True

        # Rerender foreground when requested
        if self._needs_fg_repaint:
            self._needs_fg_repaint = False
            self._cached_fg_img = self._rerender_fg(self._cached_bg_img.copy())

        return self._post_rerender(self._cached_fg_img)

    def _rerender_bg(self, h: int, w: int) -> ndarray:
        raise NotImplementedError(f"Must implement '_rerender_bg' function ({self})")

    def _rerender_fg(self, bg_image: ndarray) -> ndarray:
        return NotImplementedError(f"Must implement '_rerender_fg' function ({self})")

    def _post_rerender(self, image: ndarray) -> ndarray:
        return image

    # .................................................................................................................


class _RawImageWrapperElement(BaseCallback):
    """
    Basic element which is meant to wrap around raw images (e.g. numpy arrays) when provided to a UI
    Has very limited functionality as a UI element.
    """

    # .................................................................................................................

    def __init__(self, image: ndarray, min_side_length: int = 64):

        # Store image for re-use when rendering
        image_3ch = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self._full_image = image_3ch
        self._render_image = image_3ch
        self._targ_h = -1
        self._targ_w = -1

        # Set up sizing limits
        img_hw = image.shape[0:2]
        min_scale = max(min_side_length / side for side in img_hw)
        min_h, min_w = [round(min_scale * side) for side in img_hw]
        super().__init__(min_h, min_w, is_flexible_h=True, is_flexible_w=True)

    # .................................................................................................................

    def __repr__(self) -> str:
        img_h, img_w, img_ch = self._full_image.shape
        return f"Image ({img_h}x{img_w}x{img_ch})"

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Re-use rendered image if possible, otherwise re-render to target size
        if self._targ_h != h or self._targ_w != w:
            img_h, img_w = self._full_image.shape[0:2]
            scale = min(h / img_h, w / img_w)
            fill_wh = (round(scale * img_w), round(scale * img_h))
            self._render_image = cv2.resize(self._full_image, dsize=fill_wh)
            self._targ_h = h
            self._targ_w = w

        return self._render_image

    # .................................................................................................................

    def _get_width_given_height(self, h: int) -> int:
        img_h, img_w = self._full_image.shape[0:2]
        scaled_w = max(self._cb_rdr.min_w, round(img_w * h / img_h))
        return scaled_w

    def _get_height_given_width(self, w: int) -> int:
        img_h, img_w = self._full_image.shape[0:2]
        scaled_h = max(self._cb_rdr.min_h, round(img_h * w / img_w))
        return scaled_h

    def _get_height_and_width_without_hint(self) -> HWPX:
        img_h, img_w = self._full_image.shape[0:2]
        return img_h, img_w

    # .................................................................................................................
