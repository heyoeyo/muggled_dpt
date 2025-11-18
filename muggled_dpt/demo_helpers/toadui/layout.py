#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from .base import BaseCallback
from .helpers.text import TextDrawer
from .helpers.images import blank_image
from .helpers.styling import UIStyle
from .helpers.colors import interpret_coloru8, pick_contrasting_gray_color

# For type hints
from typing import Iterable, Any
from numpy import ndarray
from .base import BaseOverlay, CBRenderSizing
from .helpers.types import SelfType, HWPX, COLORU8


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class HStack(BaseCallback):
    """Layout element which stacks UI items together horizontally"""

    # .................................................................................................................

    def __init__(
        self,
        *items: BaseCallback,
        flex: Iterable[float | None] | None = None,
        min_w: int | None = None,
        error_on_size_constraints: bool = False,
    ):

        # Inherit from parent, with dummy values (needed so we can get child iterator!)
        super().__init__(32, 32)
        self._append_cb_children(*items)

        # Use child sizing to determine stack sizing
        tallest_child_min_h = max(child._cb_rdr.min_h for child in self)
        total_child_min_w = sum(child._cb_rdr.min_w for child in self)
        is_flex_h = any(child._cb_rdr.is_flexible_h for child in self)
        is_flex_w = any(child._cb_rdr.is_flexible_w for child in self)
        min_h = tallest_child_min_h
        min_w = max(total_child_min_w, min_w if min_w is not None else 1)
        self._cb_rdr = CBRenderSizing(min_h, min_w, is_flex_h, is_flex_w)

        # Default to sizing by aspect ratio if no flex values are given
        # -> Don't use AR sizing if not flexible (implies stacking doesn't have target AR)
        multiple_ar_children = sum(child._get_dynamic_aspect_ratio() is not None for child in self) > 1
        self._size_by_ar = flex is None and (is_flex_w and is_flex_h) and multiple_ar_children

        # Pre-compute sizing info for handling flexible sizing
        is_flex_w_per_child = [child._cb_rdr.is_flexible_w for child in self]
        flex_per_child_list = _read_flex_values(is_flex_w_per_child, flex)
        fixed_width_of_children = 0
        for child, flex_val in zip(self, flex_per_child_list):
            is_flexible = flex_val > 1e-2
            fixed_width_of_children += 0 if is_flexible else child._cb_rdr.min_w
        self._fixed_flex_width = fixed_width_of_children
        self._cumlative_flex = np.cumsum(flex_per_child_list, dtype=np.float32)
        self._flex_debug = flex_per_child_list
        self._error_on_constraints = error_on_size_constraints

        # Set up element styling when having to pad child items
        self.style = UIStyle(
            pad_color=(0, 0, 0),
            pad_border_type=cv2.BORDER_CONSTANT,
        )

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Set up starting stack point, used to keep track of child callback regions
        x_stack = self._cb_region.x1
        y_stack = self._cb_region.y1

        if self._size_by_ar:
            w_per_child_list = [child._get_width_given_height(h) for child in self]
            total_w = sum(w_per_child_list)
            if total_w != w:
                fix_list = []
                flex_list = []
                for child, child_w in zip(self, w_per_child_list):
                    fix_list.append(0 if child._cb_rdr.is_flexible_w else child._cb_rdr.min_w)
                    flex_list.append(child_w if child._cb_rdr.is_flexible_w else 0)

                avail_w = w - sum(fix_list)
                cumulative_w = np.cumsum(flex_list, dtype=np.float32) * avail_w / sum(flex_list)
                flex_list = np.diff(np.int32(np.round(cumulative_w)), prepend=0).tolist()
                w_per_child_list = [fixed_w if fixed_w > 0 else flex_w for fixed_w, flex_w in zip(fix_list, flex_list)]
        else:
            # Assign per-element sizing, taking into account flex scaling
            avail_w = max(0, w - self._fixed_flex_width)
            flex_w_px = np.diff(np.int32(np.round(self._cumlative_flex * avail_w)), prepend=0).tolist()
            w_per_child_list = [flex_w if flex_w > 0 else child._cb_rdr.min_w for child, flex_w in zip(self, flex_w_px)]

        # Have each child item draw itself
        imgs_list = []
        for child, ch_render_w in zip(self, w_per_child_list):
            frame = child._render_up_to_size(h, ch_render_w)
            frame_h, frame_w = frame.shape[0:2]

            # Crop overly-tall images
            # -> Don't need to crop wide images, since h-stacking won't break!
            if frame_h > h:
                print(
                    f"Render sizing error! Expecting height: {h}, got {frame_h} ({child})",
                    "-> Will crop!",
                    sep="\n",
                )
                frame = frame[:h, :, :]
                frame_h = h

            # Adjust frame height if needed
            tpad, lpad, bpad, rpad = 0, 0, 0, 0
            need_pad = (frame_h < h) or (frame_w < ch_render_w)
            if need_pad:
                available_h = h - frame_h
                available_w = max(0, ch_render_w - frame_w)
                tpad, lpad = available_h // 2, available_w // 2
                bpad, rpad = available_h - tpad, available_w - lpad
                ptype = self.style.pad_border_type
                pcolor = self.style.pad_color
                frame = cv2.copyMakeBorder(frame, tpad, bpad, lpad, rpad, ptype, value=pcolor)

            # Store image
            imgs_list.append(frame)

            # Provide callback region to child item
            x1, y1 = x_stack + lpad, y_stack + tpad
            x2, y2 = x1 + frame_w, y1 + frame_h
            child._update_cb_region(x1, y1, x2, y2)

            # Update stacking point for next child
            x_stack = x2 + rpad

        return np.hstack(imgs_list)

    # .................................................................................................................

    def _get_dynamic_aspect_ratio(self) -> float | None:
        if self._cb_rdr.is_flexible_h and self._cb_rdr.is_flexible_w:
            child_ar = (c._get_dynamic_aspect_ratio() for c in self)
            ar = sum(ar if ar is not None else 0 for ar in child_ar)
            return None if ar == 0 else ar
        return None

    def _get_width_given_height(self, h: int) -> int:

        if not self._cb_rdr.is_flexible_w:
            return self._cb_rdr.min_w

        # Ask child elements for desired width and use total
        w_per_child_list = [child._get_width_given_height(h) for child in self]
        total_child_w = sum(w_per_child_list)
        return max(self._cb_rdr.min_w, total_child_w)

    def _get_height_given_width(self, w: int) -> int:
        """
        For h-stacking, we normally want to set a height since this must be shared
        for all elements in order to stack horizontally. Here we don't know the height.

        If sizing by aspect ratio, we calculate the height from knowing that all
        items must stack to the target width, while sharing the same height:
            target_w = (h * ar1) + (h * ar2) + (h * ar3) + ...
            target_w = h * (ar1 + ar2 + ar3)
            Therefore, h = target_w / sum(ar for all item aspect ratios)

        If sizing by flex values, we first figure out how much 'width' each child
        should be assigned. Then each child is asked for it's render height, given
        the assigned width. We take the 'tallest' child height as the height for stacking.

        Returns:
            render_height
        """

        # Use fixed height if not flexible
        if not self._cb_rdr.is_flexible_h:
            return self._cb_rdr.min_h

        # Allocate height based on child aspect ratios
        if self._size_by_ar:
            avail_w = max(0, w - self._fixed_flex_width)
            child_ar = (c._get_dynamic_aspect_ratio() for c in self)
            h = avail_w / sum(ar if ar is not None else 0 for ar in child_ar)
            return max(self._cb_rdr.min_h, round(h))

        # Figure out per-element width based on flex assignment
        avail_w = max(0, w - self._fixed_flex_width)
        flex_sizing_px = np.diff(np.int32(np.round(self._cumlative_flex * avail_w)), prepend=0).tolist()
        w_per_child_list = [max(child._cb_rdr.min_w, flex_w) for child, flex_w in zip(self, flex_sizing_px)]

        # Sanity check
        if self._error_on_constraints:
            total_computed_w = sum(w_per_child_list)
            assert total_computed_w == w, f"Error computing target widths ({self})! Target: {w}, got {total_computed_w}"

        # We'll say our height, given the target width, is that of the tallest child element
        h_per_child_list = (child._get_height_given_width(w) for child, w in zip(self, w_per_child_list))
        return max(h_per_child_list)

    # .................................................................................................................


class VStack(BaseCallback):
    """Layout element which stacks UI items together vertically"""

    # .................................................................................................................

    def __init__(
        self,
        *items: BaseCallback | None,
        flex: Iterable[float | None] | None = None,
        min_h: int | None = None,
        error_on_size_constraints: bool = False,
    ):

        # Inherit from parent, with dummy values (needed so we can get child iterator!)
        super().__init__(32, 32)
        self._append_cb_children(*items)

        # Update stack sizing based on children
        total_child_min_h = sum(child._cb_rdr.min_h for child in self)
        widest_child_min_w = max(child._cb_rdr.min_w for child in self)
        is_flex_h = any(child._cb_rdr.is_flexible_h for child in self)
        is_flex_w = any(child._cb_rdr.is_flexible_w for child in self)
        min_h = max(total_child_min_h, min_h if min_h is not None else 1)
        min_w = widest_child_min_w
        self._cb_rdr = CBRenderSizing(min_h, min_w, is_flex_h, is_flex_w)

        # Default to sizing by aspect ratio if not given flex values
        # -> Don't use AR sizing if not flexible (implies stacking doesn't have target AR)
        multiple_ar_children = sum(child._get_dynamic_aspect_ratio() is not None for child in self) > 1
        self._size_by_ar = flex is None and (is_flex_w and is_flex_h) and multiple_ar_children

        # Pre-compute sizing info for handling flexible sizing
        is_flex_h_per_child = [child._cb_rdr.is_flexible_h for child in self]
        flex_per_child_list = _read_flex_values(is_flex_h_per_child, flex)
        fixed_height_of_children = 0
        for child, flex_val in zip(self, flex_per_child_list):
            is_flexible = flex_val > 1e-3
            fixed_height_of_children += 0 if is_flexible else child._cb_rdr.min_h
        self._fixed_flex_height = fixed_height_of_children
        self._cumlative_flex = np.cumsum(flex_per_child_list, dtype=np.float32)
        self._flex_debug = flex_per_child_list
        self._error_on_constraints = error_on_size_constraints

        # Set up element styling when having to pad child items
        self.style = UIStyle(
            pad_color=(0, 0, 0),
            pad_border_type=cv2.BORDER_CONSTANT,
        )

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Set up starting stack point, used to keep track of child callback regions
        x_stack = self._cb_region.x1
        y_stack = self._cb_region.y1

        if self._size_by_ar:
            h_per_child_list = [child._get_height_given_width(w) for child in self]
            total_h = sum(h_per_child_list)
            if total_h != h:
                fix_list = []
                flex_list = []
                for child, child_h in zip(self, h_per_child_list):
                    fix_list.append(0 if child._cb_rdr.is_flexible_h else child._cb_rdr.min_h)
                    flex_list.append(child_h if child._cb_rdr.is_flexible_h else 0)

                avail_h = h - sum(fix_list)
                cumulative_h = np.cumsum(flex_list, dtype=np.float32) * avail_h / sum(flex_list)
                flex_list = np.diff(np.int32(np.round(cumulative_h)), prepend=0).tolist()
                h_per_child_list = [fixed_h if fixed_h > 0 else flex_h for fixed_h, flex_h in zip(fix_list, flex_list)]
        else:
            # Assign per-element sizing, taking into account flex scaling
            avail_h = max(0, h - self._fixed_flex_height)
            flex_h_px = np.diff(np.int32(np.round(self._cumlative_flex * avail_h)), prepend=0).tolist()
            h_per_child_list = [flex_h if flex_h > 0 else child._cb_rdr.min_h for child, flex_h in zip(self, flex_h_px)]

        # Have each child item draw itself
        imgs_list = []
        for child, ch_render_h in zip(self, h_per_child_list):
            frame = child._render_up_to_size(ch_render_h, w)
            frame_h, frame_w = frame.shape[0:2]

            # Crop overly-wide images
            # -> Don't need to crop tall images, since v-stacking won't break!
            if frame_w > w:
                print(
                    f"Render sizing error! Expecting width: {w}, got {frame_w} ({child})",
                    "-> Will crop!",
                    sep="\n",
                )
                frame = frame[:, :w, :]
                frame_w = w

            # Adjust frame width if needed
            tpad, lpad, bpad, rpad = 0, 0, 0, 0
            need_pad = (frame_w < w) or (frame_h < ch_render_h)
            if need_pad:
                available_w = w - frame_w
                available_h = max(0, ch_render_h - frame_h)
                tpad, lpad = available_h // 2, available_w // 2
                bpad, rpad = available_h - tpad, available_w - lpad
                ptype = self.style.pad_border_type
                pcolor = self.style.pad_color
                frame = cv2.copyMakeBorder(frame, tpad, bpad, lpad, rpad, ptype, value=pcolor)
                # print(" vpad->", tpad, bpad, lpad, rpad)

            # Store image
            imgs_list.append(frame)

            # Provide callback region to child item
            x1, y1 = x_stack + lpad, y_stack + tpad
            x2, y2 = x1 + frame_w, y1 + frame_h
            child._update_cb_region(x1, y1, x2, y2)

            # Update stacking point for next child
            y_stack = y2 + bpad

        return np.vstack(imgs_list)

    # .................................................................................................................

    def _get_dynamic_aspect_ratio(self) -> float | None:
        if self._cb_rdr.is_flexible_h and self._cb_rdr.is_flexible_w:
            child_ar = (c._get_dynamic_aspect_ratio() for c in self)
            inv_ar_sum = sum(1 / ar if ar is not None else 0 for ar in child_ar)
            return None if inv_ar_sum == 0 else 1 / inv_ar_sum

        return None

    def _get_height_given_width(self, w: int) -> int:

        # Use fixed height if not flexible
        if not self._cb_rdr.is_flexible_h:
            return self._cb_rdr.min_h

        # Ask child elements for desired width and use total
        h_per_child_list = [child._get_height_given_width(w) for child in self]
        total_child_h = sum(h_per_child_list)
        return max(self._cb_rdr.min_h, total_child_h)

    def _get_width_given_height(self, h: int) -> int:
        """
        For v-stacking, we normally want to set a width since this must be shared
        for all elements in order to stack vertically. Here we don't know the width.

        If sizing by aspect ratio, we calculate the width from knowing that all
        items must stack to the target height, while sharing the same width:
            target_h = (w / ar1) + (w / ar2) + (w / ar3) + ...
            target_h = w * (1/ar1 + 1/ar2 + 1/ar3)
            Therefore, w = target_h / sum(1/ar for all item aspect ratios)

        If sizing by flex values, we first figure out how much 'height' each child
        should be assigned. Then each child is asked for it's render width, given
        the assigned height. We take the 'widest' child width as the width for stacking.

        Returns:
            render_width
        """

        # Use fixed width if not flexible
        if not self._cb_rdr.is_flexible_w:
            return self._cb_rdr.min_w

        # Allocate width based on child aspect ratios
        if self._size_by_ar:
            avail_h = max(1, h - self._fixed_flex_height)
            child_ar = (c._get_dynamic_aspect_ratio() for c in self)
            w = avail_h / sum(1 / ar if ar is not None else 0 for ar in child_ar)
            return max(self._cb_rdr.min_w, round(w))

        # Figure out per-element height based on flex assignment
        avail_h = max(0, h - self._fixed_flex_height)
        flex_sizing_px = np.diff(np.int32(np.round(self._cumlative_flex * avail_h)), prepend=0).tolist()
        h_per_child_list = [max(child._cb_rdr.min_h, flex_h) for child, flex_h in zip(self, flex_sizing_px)]

        # Sanity check
        if self._error_on_constraints:
            total_computed_h = sum(h_per_child_list)
            assert total_computed_h == h, f"Error computing target height ({self})! Target: {h}, got {total_computed_h}"

        # We'll say our width, given the target height, is that of the widest child element
        w_per_child_list = (child._get_width_given_height(h=h) for child, h in zip(self, h_per_child_list))
        return max(w_per_child_list)

    # .................................................................................................................


class GridStack(BaseCallback):
    """
    Layout which combines elements into a grid with a specified number of rows and columns.
    This is a convenience wrapper around a combination of:
        VStack(HStack(...), HStack(...), ...) (row ordered)
        or
        HStack(VStack(...), VStack(...), ...) (column ordered)
    """

    # .................................................................................................................

    def __init__(
        self,
        *items,
        num_rows: int | None = None,
        num_columns: int | None = None,
        target_aspect_ratio: float = 1,
        is_row_ordered: bool = True,
    ):

        # Inherit from parent
        super().__init__(128, 128)

        # Fill in missing row/column counts
        self._items = tuple(items)
        num_items = len(self._items)
        if num_rows is None and num_columns is None:
            num_rows, num_columns = self.get_row_column_by_aspect_ratio(num_items, target_aspect_ratio)
        elif num_rows is None:
            num_rows = int(np.ceil(num_items / num_columns))
        elif num_columns is None:
            num_columns = int(np.ceil(num_items / num_rows))
        self._num_rows = num_rows
        self._num_cols = num_columns

        # Store ordering info
        self._is_row_ordered = is_row_ordered
        self._layout: VStack | HStack = self._build_new_layout()

        # Set up element styling when having to pad child items
        self.style = UIStyle(
            pad_color=(0, 0, 0),
            pad_border_type=cv2.BORDER_CONSTANT,
        )

    # .................................................................................................................

    def get_num_rows_columns(self) -> tuple[int, int]:
        """Get current row/column count of the grid layout"""
        return (self._num_rows, self._num_cols)

    def set_num_rows_columns(self, num_rows: int, num_columns: int) -> SelfType:
        """Update the number of rows & columns of the grid"""
        self._num_rows = num_rows
        self._num_cols = num_columns
        return self

    def increment_num_rows(self, increment: int = 1) -> SelfType:
        """Change the grid arrangement by increasing the number of rows"""

        num_items = len(self)
        rowcol_options = self.get_row_column_options(num_items)
        curr_rowcol = self.get_num_rows_columns()

        # Figure out which of the options we are currently using (or close to)
        curr_opt_idx = rowcol_options.index(curr_rowcol) if curr_rowcol in rowcol_options else None
        if curr_opt_idx is None:
            possible_row_indices = [row_idx for row_idx, _ in rowcol_options if row_idx <= curr_rowcol[0]]
            curr_opt_idx = max(0, len(possible_row_indices) - 1)

        # Use the next row/column count
        next_opt_idx = (curr_opt_idx + increment) % len(rowcol_options)
        num_rows, num_cols = rowcol_options[next_opt_idx]

        # Update internal records
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._layout = self._build_new_layout()

        return self

    def decrement_num_rows(self, decrement: int = 1) -> SelfType:
        """Change the grid arrangement by decreasing the number of rows"""
        return self.increment_num_rows(-decrement)

    # .................................................................................................................

    def transpose(self) -> SelfType:
        """Flip number of rows & columns"""
        self._num_rows, self._num_cols = self._num_cols, self._num_rows
        self._is_row_ordered = not self._is_row_ordered
        self._layout = self._build_new_layout()
        return self

    # .................................................................................................................

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, index: int) -> BaseCallback:
        return self._items[index]

    def row_iter(self) -> tuple[int, tuple]:
        """
        Iterator over items per row. Example:
            for row_idx, items_in_row in grid.row_iter():
                # ... Do something with each row ...

                for col_idx, item for enumerate(items_in_row):
                    # ... Do something with each item ...
                    pass
                pass
        """

        for row_idx in range(self._num_rows):
            idx1 = row_idx * self._num_cols
            idx2 = idx1 + self._num_cols
            items_in_row = self[idx1:idx2]
            if len(items_in_row) == 0:
                break
            yield row_idx, tuple(items_in_row)

        return

    def column_iter(self) -> tuple[int, tuple]:
        """
        Iterator over items per column. Example:
            for col_idx, items_in_column in grid.column_iter():
                # ... Do something with each column ...

                for row_idx, item for enumerate(items_in_column):
                    # ... Do something with each item ...
                    pass
                pass
        """

        num_items = len(self._items)
        for col_idx in range(self._num_cols):
            item_idxs = [col_idx + row_idx * self._num_cols for row_idx in range(self._num_rows)]
            items_in_column = tuple(self[item_idx] for item_idx in item_idxs if item_idx < num_items)
            if len(items_in_column) == 0:
                break
            yield col_idx, items_in_column

        return

    def grid_iter(self) -> tuple[int, int, BaseCallback]:
        """
        Iterator over all items while returning row/column index. Example:
            for row_idx, col_idx, item in grid.grid_iter():
                # ... Do something with each item ...
                pass
        """

        for item_idx, item in enumerate(self):
            row_idx = item_idx // self._num_cols
            col_idx = item_idx % self._num_cols
            yield row_idx, col_idx, item

        return

    # .................................................................................................................

    def _render_up_to_size(self, h, w):
        outimg = self._layout._render_up_to_size(h, w)
        self._cb_region = self._layout._cb_region
        return outimg

    def _get_dynamic_aspect_ratio(self):
        return self._layout._get_dynamic_aspect_ratio()

    def _get_height_and_width_without_hint(self):
        return self._layout._get_height_and_width_without_hint()

    def _get_height_given_width(self, w):
        return self._layout._get_height_given_width(w)

    def _get_width_given_height(self, h):
        return self._layout._get_width_given_height(h)

    def _update_cb_region(self, x1, y1, x2, y2) -> SelfType:
        self._layout._update_cb_region(x1, y1, x2, y2)
        self._cb_region = self._layout._cb_region
        return

    def _build_new_layout(self) -> VStack | HStack:
        """
        Helper used to re-build the internal V/HStack layout structure,
        needed when the number of rows or columns changes
        """

        # Clear existing layout parent relationship from items (we're about to replace it)
        try:
            for item in self._items:
                item._clear_cb_parent(self._layout)
        except AttributeError:
            # Expected to happen on first run, since there is no 'self._layout' built yet!
            # -> This is ok to skip if we haven't set up a parent relationship on the items
            pass

        # Set up parameters used to build layout for row-vs-column ordering
        # - Default to stacking horizontally (axis) and then vertical (alt)
        # - 'axis' refers to the main axis (i.e. items 0, 1, 2, etc. are stacked into first)
        # - 'alt' refers to the alternate axis (i.e. stacks of [0, 1, 2], [4, 5, 6], etc.)
        axis_alt_count = (self._num_cols, self._num_rows)
        axis_alt_stack = (HStack, VStack)
        if not self._is_row_ordered:
            axis_alt_count = tuple(reversed(axis_alt_count))
            axis_alt_stack = tuple(reversed(axis_alt_stack))

        # Build vertical stack of rows (row-order) or horizontal stack of column (column-order) layout
        axis_count, alt_count = axis_alt_count
        axis_stack, alt_stack = axis_alt_stack
        axis_stacks_list = []
        for axis_idx in range(alt_count):
            idx_offset = axis_idx * axis_count
            item_slice = slice(idx_offset, idx_offset + axis_count)
            item_list = self._items[item_slice]
            axis_stacks_list.append(axis_stack(*item_list))
        new_layout = alt_stack(*axis_stacks_list)  # This is: VStack(*[HStack, ...]) or HStack(*[VStack, ...])

        # Update sizing constraints
        min_h_list = [axis._cb_rdr.min_h for axis in new_layout]
        min_w_list = [axis._cb_rdr.min_w for axis in new_layout]
        is_flex_h = any(axis._cb_rdr.is_flexible_h for axis in new_layout)
        is_flex_w = any(axis._cb_rdr.is_flexible_w for axis in new_layout)
        min_h = sum(min_h_list) if self._is_row_ordered else max(min_h_list)
        min_w = max(min_w_list) if self._is_row_ordered else sum(min_w_list)
        self._cb_rdr = CBRenderSizing(min_h, min_w, is_flex_h, is_flex_w)

        # Use new layout as (only) child element
        self._clear_cb_children()
        self._append_cb_children(new_layout)

        return new_layout

    # .................................................................................................................

    @staticmethod
    def get_row_column_options(num_items: int, include_all_row_counts=False) -> tuple[tuple[int, int]]:
        """
        Helper used to get all possible neatly divisible combinations of (num_rows, num_columns)
        for a given number of items, in order of fewest rows -to- most rows.
        For example for num_items = 6, returns:
            ((1, 6), (2, 3), (3, 2), (6, 1))
            -> This is meant to be interpreted as:
                (1 row, 6 columns) OR (2 rows, 3 columns) OR (3 rows, 2 columns) OR (6 rows, 1 column)

        As another example, for num_items = 12, returns:
            ((1, 12), (2, 6), (3, 4), (4, 3), (6, 2), (12, 1))

        If 'include_all_row_counts' is True, the all possible 'number of rows' will be considered.
        This can lead to a total number of grid cells which exceeds the number of items!
        For example, for num_items = 5, normally the result would be:
            ((1, 5), (5, 1))
        But with include_all_row_counts=True, the result is instead:
            ((1, 5), (2, 3), (3, 2), (4, 2), (5, 1))
        Note that all row counts are present (1, 2, 3, 4, 5), but some (e.g. (2, 3) or (4, 2))
        result in a total number of cells (6 and 8 respectively) which exceed the item count of 5!
        """
        if include_all_row_counts:
            return tuple((k, int(np.ceil(num_items / k))) for k in range(1, 1 + num_items))
        return tuple((k, num_items // k) for k in range(1, 1 + num_items) if (num_items % k) == 0)

    # .................................................................................................................

    @staticmethod
    def get_aspect_ratio_similarity(
        row_column_options: tuple[tuple[int, int]], target_aspect_ratio: float
    ) -> list[float]:
        """
        Compute similarity score (0 to 1) indicating how close of match
        each row/column option is to the target aspect ratio.

        Note that the row_column_options are expected to come from the
        .get_row_column_options(...) method
        """
        target_theta, pi_over_2 = np.arctan(target_aspect_ratio), np.pi / 2
        difference_scores = (abs(np.arctan(col / row) - target_theta) for row, col in row_column_options)
        return tuple(float(1.0 - (diff / pi_over_2)) for diff in difference_scores)

    # .................................................................................................................

    @classmethod
    def get_row_column_by_aspect_ratio(cls, num_items: int, target_aspect_ratio: float = 1.0) -> tuple[int, int]:
        """
        Helper used to choose the number of rows & columns to best match a target aspect ratio
        Returns: (num_rows, num_columns)
        """

        rc_options = cls.get_row_column_options(num_items)
        ar_similarity = cls.get_aspect_ratio_similarity(rc_options, target_aspect_ratio)
        best_match_idx = np.argmax(ar_similarity)

        return rc_options[best_match_idx]

    # .................................................................................................................


class OverlayStack(BaseCallback):
    """
    Element used to combine multiple overlays onto a single base item.
    (i.e. stacks overlays ontop of one another).

    This is mainly intended for better efficiency when using many overlays together.
    """

    # .................................................................................................................

    def __init__(self, base_item: BaseCallback, *overlay_items: BaseOverlay, suppress_callbacks_to_base: bool = False):

        # Inherit from parent and copy base item render limits
        super().__init__(32, 32)

        # Clear overlay children, so we don't get duplicate base item calls
        # (only this parent instance will handle callbacks & pass these down to children)
        self._cb_rdr = base_item._cb_rdr.copy()
        for olay in overlay_items:
            olay._cb_rdr = base_item._cb_rdr.copy()
            olay._cb_child_list.clear()

        # Store base item for re-use in rendering and attach as child if we want to pass callbacks to it
        self._base_item = base_item
        if not suppress_callbacks_to_base:
            self._append_cb_children(self._base_item)

        # Store initial overlays
        self._overlay_items = tuple()
        self.add_overlays(*overlay_items)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        olay_names = [str(olay) for olay in self._overlay_items]
        return f"{cls_name} [{self._base_item} | {', '.join(olay_names)}]"

    # .................................................................................................................

    def add_overlays(self, *overlay_items: BaseOverlay) -> SelfType:
        """Function used to add overlays"""

        # Attach child overlays
        # - Try to (forcefully) give them a copy of the base item (if they have existing storage for it)
        # - This allows overlays to directly reference the base item from user code
        # - This isn't needed for the overlay stack itself (just a QoL feature)
        new_olay_items = tuple(overlay_items)
        for olay_item in new_olay_items:
            if hasattr(olay_item, "_base_item"):
                olay_item._base_item = self._base_item
            self._append_cb_children(olay_item)

        # Update storage of overlays to include new items
        olays_list = list(self._overlay_items)
        olays_list.extend(new_olay_items)
        self._overlay_items = tuple(olays_list)

        return self

    def get_base_item(self) -> BaseCallback:
        """Function used to get access to the underlying base element"""
        return self._base_item

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Set up starting stack point, used to keep track of child callback regions
        x1 = self._cb_region.x1
        y1 = self._cb_region.y1

        # Have base item provide the base frame rendering and overlays handle drawing over-top
        base_frame = self._base_item._render_up_to_size(h, w).copy()
        base_h, base_w = base_frame.shape[0:2]

        x2, y2 = x1 + base_w, y1 + base_h
        self._base_item._update_cb_region(x1, y1, x2, y2)
        for overlay in self._overlay_items:
            overlay._update_cb_region(x1, y1, x2, y2)
            base_frame = overlay._render_overlay(base_frame) if overlay._enable_overlay_render else base_frame

        return base_frame

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


class Swapper(BaseCallback):
    """
    Special layout item which allows for swapping between elements.
    This can be used to switch between different UI layouts, for example.

    Items can be swapped between using .set_swap_index(...), which
    will switch to items based on the indexing order used when
    initializing the swap instance.
    Alternatively, an optional 'keys' init argument can be used
    to assign labels, which can be swapped between using
    the .set_swap_key(...) function.
    """

    # .................................................................................................................

    def __init__(self, *swap_items: BaseCallback, initial_index: int = 0, keys: Iterable[Any] | None = None):

        # Fill in missing keys
        swap_items = tuple(swap_items)
        if keys is None:
            keys = range(len(swap_items))
        keys = tuple(keys)

        # Set up per-item/key removal flag
        is_none_keys = [k is None for k in keys]
        is_none_items = [item is None for item in swap_items]
        need_remove_list = [any(none_key_or_item) for none_key_or_item in zip(is_none_keys, is_none_items)]
        items = tuple(item for item, remove in zip(swap_items, need_remove_list) if not remove)
        keys = tuple(key for key, remove in zip(keys, need_remove_list) if not remove)

        # Sanity check
        num_keys, num_items = len(keys), len(items)
        assert len(set(keys)) == num_keys, f"Cannot have duplicate keys: {keys}"
        assert num_keys == num_items, f"Number of keys ({num_keys}) must match number of swap items ({num_items})"

        # Inherit from parent
        initial_index = max(0, min(num_items - 1, initial_index))
        item_rdr = items[initial_index]._cb_rdr
        super().__init__(item_rdr.min_h, item_rdr.min_w, item_rdr.is_flexible_h, item_rdr.is_flexible_w)

        # Store items for swapping
        self._is_changed = True
        self._items: tuple[BaseCallback] = items
        self._num_items: int = len(self._items)
        self._key_to_idx_lut = {key: idx for idx, key in enumerate(keys)}

        # Set up initial swap item
        self._swap_idx: int = initial_index
        self._active_item: BaseCallback = self._items[initial_index]
        self._cb_rdr: CBRenderSizing = self._active_item._cb_rdr.copy()
        self._append_cb_children(self._active_item)

    # .................................................................................................................

    def get_active_element(self) -> BaseCallback:
        """Helper used to access the currently active swap item (without reading or consuming 'is changed')"""
        return self._active_item

    def read(self) -> tuple[bool, int, BaseCallback]:
        """Get a reference to the current swap element. Returns: is_changed, swap_index, swap_element"""
        is_changed, self._is_changed = self._is_changed, False
        return is_changed, self._swap_idx, self._active_item

    # .................................................................................................................

    def set_swap_index(self, swap_index: int) -> BaseCallback:
        """
        Swap to a new item, by index
        Returns:
            current_swap_item
        """

        if swap_index < self._num_items:
            # Handle negative indexing
            if swap_index < 0:
                swap_index = max(0, self._num_items - swap_index)

            # Only update if index is actually different (don't want to trigger change event otherwise)
            prev_idx = self._swap_idx
            if swap_index != prev_idx:

                # Clean up callback handling (want active element to seem like the only child callback)
                # -> Also want to clear self as parent before re-appending child, so we don't have repeats!
                self._clear_cb_children()
                new_active_element = self._items[swap_index]
                new_active_element._clear_cb_parent(self)
                self._append_cb_children(new_active_element)

                # Update internal state tracking
                self._swap_idx = swap_index
                self._active_item = new_active_element
                self._cb_rdr = self._active_item._cb_rdr.copy()
                self._is_changed = True
            pass

        return self._active_item

    def set_swap_key(self, swap_key: Any) -> BaseCallback:
        """
        Swap to a new item, by key name. Keys can be specified on init.
        Returns:
            current_swap_item
        """
        new_idx = self._key_to_idx_lut[swap_key]
        return self.set_swap_index(new_idx)

    # .................................................................................................................

    def next(self, increment=1) -> BaseCallback:
        new_idx = (self._swap_idx + increment) % self._num_items
        return self.set_swap_index(new_idx)

    def prev(self, decrement=1) -> BaseCallback:
        return self.next(-decrement)

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:
        return self._active_item._render_up_to_size(h, w)

    def _get_dynamic_aspect_ratio(self):
        return self._active_item._get_dynamic_aspect_ratio()

    def _get_height_and_width_without_hint(self):
        return self._active_item._get_height_and_width_without_hint()

    def _get_height_given_width(self, w):
        return self._active_item._get_height_given_width(w)

    def _get_width_given_height(self, h):
        return self._active_item._get_width_given_height(h)

    def _update_cb_region(self, x1, y1, x2, y2) -> SelfType:
        self._active_item._update_cb_region(x1, y1, x2, y2)
        self._cb_region = self._active_item._cb_region
        return self

    # .................................................................................................................


class HSeparator(BaseCallback):
    """Simple element used to create a horizontal space between other elements"""

    # .................................................................................................................

    def __init__(
        self,
        width: int = 2,
        color: COLORU8 | int = (20, 20, 20),
        label: str | None = None,
        is_flexible_h: bool = True,
        is_flexible_w: bool = False,
    ):
        self._cached_img = blank_image(1, width, color)
        self._label = label
        self.style = UIStyle(
            color=interpret_coloru8(color),
            text=None if label is None else TextDrawer(0.35, 1, pick_contrasting_gray_color(color)),
        )
        super().__init__(1, width, is_flexible_h=is_flexible_h, is_flexible_w=is_flexible_w)

    # .................................................................................................................

    @classmethod
    def many(cls, num_separators: int, width: int = 2, color: COLORU8 | int = (20, 20, 20)):
        return [cls(width, color) for _ in range(num_separators)]

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:
        img_h, img_w = self._cached_img.shape[0:2]
        if img_h != h or img_w != w:
            self._cached_img = blank_image(h, w, self.style.color)
            if self._label is not None:
                self._cached_img = self.style.text.xy_centered(self._cached_img, self._label)
        return self._cached_img

    # .................................................................................................................


class VSeparator(BaseCallback):
    """Simple element used to create a vertical space between other elements"""

    # .................................................................................................................

    def __init__(
        self,
        height: int = 2,
        color: COLORU8 | int = (20, 20, 20),
        label: str | None = None,
        is_flexible_h: bool = False,
        is_flexible_w: bool = True,
    ):
        self._cached_img = blank_image(height, 1, color)
        self._label = label
        self.style = UIStyle(
            color=interpret_coloru8(color),
            text=None if label is None else TextDrawer(0.35, 1, pick_contrasting_gray_color(color)),
        )
        super().__init__(height, 1, is_flexible_h=is_flexible_h, is_flexible_w=is_flexible_w)

    # .................................................................................................................

    @classmethod
    def many(cls, num_separators: int, height: int = 2, color: COLORU8 | int = (20, 20, 20)):
        return [cls(height, color) for _ in range(num_separators)]

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:
        img_h, img_w = self._cached_img.shape[0:2]
        if img_h != h or img_w != w:
            self._cached_img = blank_image(h, w, self.style.color)
            if self._label is not None:
                self._cached_img = self.style.text.xy_centered(self._cached_img, self._label)
        return self._cached_img

    # .................................................................................................................


class Padded(BaseCallback):
    """
    Simple element used to add padding around another element
    Padding can be provided as 4 numbers, 2 numbers or 1 number:
        - If 4 numbers are given, interpret as: (x1, y1, x2, y2)
        - If 2 numbers, interpret as: (x1 & x2, y1 & y2)
          -> For example: (24, 0) gives 24px padding to left & right and 0 to top & bottom
        - If 1 number is given, all sides get the same padding

    Also supports adding labels to the top/bottom segments
    """

    # .................................................................................................................

    def __init__(
        self,
        base_item: BaseCallback,
        pad_px: tuple[int, int, int, int] | tuple[int, int] | int = 8,
        color: COLORU8 | int = (20, 20, 20),
        top_label: str | None = None,
        bottom_label: str | None = None,
        inner_outline_color: COLORU8 | int | None = None,
        outer_outline_color: COLORU8 | int | None = None,
    ):
        # Handle variety of ways of providing top/right/bottom/left inputs
        if not isinstance(pad_px, Iterable):
            pad_px = tuple(int(pad_px) for _ in range(4))
        if len(pad_px) == 2:
            x_pad, y_pad = pad_px
            pad_px = (x_pad, y_pad, x_pad, y_pad)
        assert len(pad_px) == 4, "Bad padding format! Must be given as 1, 2 or 4 integers"
        pad_px = tuple(max(0, int(size)) for size in pad_px)

        # Allocate storage for cached padding image components
        pad_color = interpret_coloru8(color)
        self._label_t = top_label
        self._label_b = bottom_label
        self._cache_t = blank_image(0, 0, pad_color)
        self._cache_b = blank_image(0, 0, pad_color)
        self._cache_l = blank_image(0, 0, pad_color)
        self._cache_r = blank_image(0, 0, pad_color)
        self._xy1xy2_pad_px = pad_px
        self._cached_hw = (-1, -1)
        self._base_item = base_item

        # Set up element styling
        lpad, tpad, rpad, bpad = pad_px
        self.style = UIStyle(
            color=pad_color,
            text_top=TextDrawer(0.35, 1, pick_contrasting_gray_color(color), max_height=tpad),
            text_bottom=TextDrawer(0.35, 1, pick_contrasting_gray_color(color), max_height=bpad),
            top_label_xy_norm=(0.5, 0.5),
            top_label_anchor_xy_norm=None,
            top_label_offset_xy_px=(0, 0),
            bottom_label_xy_norm=(0.5, 0.5),
            bottom_label_anchor_xy_norm=None,
            bottom_label_offset_xy_px=(0, 0),
            color_inner_outline=interpret_coloru8(inner_outline_color),
            color_outer_outline=interpret_coloru8(outer_outline_color),
            thickness_inner_outline=1,
            thickness_outer_outline=1,
        )

        # Inherit from parent
        base_rdr = base_item._cb_rdr.copy()
        min_h = base_rdr.min_h + tpad + bpad
        min_w = base_rdr.min_h + rpad + lpad
        is_flexible_h = base_rdr.is_flexible_h
        is_flexible_w = base_rdr.is_flexible_w
        super().__init__(min_h, min_w, is_flexible_h=is_flexible_h, is_flexible_w=is_flexible_w)
        self._append_cb_children(base_item)

    # .................................................................................................................

    def force_rerender(self) -> SelfType:
        """Function used to force a re-render, which can be useful if altering styling parameters"""
        self._cached_hw = (-1, -1)
        return self

    def set_labels(self, top_label: str | None = None, bottom_label: str | None = None) -> SelfType:
        """
        Function used to update top/bottom labels
        - If 'None' is provided, the existing label will be kept as-is
        - To clear a label, provide an empty string: ''
        """

        if top_label is not None:
            top_label = str(top_label)
            self._label_t = top_label if len(top_label) > 0 else None
            self.force_rerender()

        if bottom_label is not None:
            bottom_label = str(bottom_label)
            self._label_b = bottom_label if len(bottom_label) > 0 else None
            self.force_rerender()

        return self

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Render out the base image (with space left for padding)
        lpad, tpad, rpad, bpad = self._xy1xy2_pad_px
        tb_pad, lr_pad = tpad + bpad, lpad + rpad
        target_base_h, target_base_w = max((h - tb_pad), 1), max((w - lr_pad), 1)
        base_img = self._base_item._render_up_to_size(target_base_h, target_base_w)
        base_h, base_w = base_img.shape[0:2]

        cache_h, cache_w = self._cached_hw
        if cache_h != cache_h or cache_w != base_w:
            self._cached_hw = (base_h, base_w)
            avail_tb_pad = max(h - base_h, 0)
            avail_lr_pad = max(w - base_w, 0)

            tfract = tpad / tb_pad if tb_pad > 0 else 0.5
            actual_t_pad = round(avail_tb_pad * tfract)
            actual_b_pad = avail_tb_pad - actual_t_pad

            lfract = lpad / lr_pad if lr_pad > 0 else 0.5
            actual_l_pad = round(avail_lr_pad * lfract)
            actual_r_pad = avail_lr_pad - actual_l_pad

            t_img = blank_image(actual_t_pad, w, self.style.color)
            b_img = blank_image(actual_b_pad, w, self.style.color)
            l_img = blank_image(base_h, actual_l_pad, self.style.color)
            r_img = blank_image(base_h, actual_r_pad, self.style.color)

            # Draw outlines (using filled rectangles to avoid rounded corners)
            if self.style.color_inner_outline is not None:
                ol_color, ol_thick = self.style.color_inner_outline, self.style.thickness_inner_outline
                draw_rectfill = lambda img, xy1, xy2: cv2.rectangle(img, xy1, xy2, ol_color, -1, cv2.LINE_4)
                x1, tb_x2 = actual_l_pad - ol_thick, w - actual_r_pad + ol_thick - 1
                draw_rectfill(t_img, (x1, actual_t_pad - ol_thick), (tb_x2, actual_t_pad))
                draw_rectfill(b_img, (actual_l_pad - ol_thick, 0), (tb_x2, ol_thick - 1))
                draw_rectfill(l_img, (x1, 0), (actual_l_pad, base_h))
                draw_rectfill(r_img, (0, 0), (ol_thick - 1, base_h))
            if self.style.color_outer_outline is not None:
                ol_color, ol_thick = self.style.color_outer_outline, self.style.thickness_outer_outline
                draw_rectfill = lambda img, xy1, xy2: cv2.rectangle(img, xy1, xy2, ol_color, -1, cv2.LINE_4)
                draw_rectfill(t_img, (0, 0), (w, ol_thick - 1))
                draw_rectfill(b_img, (0, actual_b_pad - ol_thick), (w, actual_b_pad))
                draw_rectfill(l_img, (0, 0), (ol_thick - 1, base_h))
                draw_rectfill(r_img, (actual_r_pad - ol_thick, 0), (actual_r_pad, base_h))

                # Draw 'left' edge that appears in the top/bottom segments
                draw_rectfill(t_img, (0, 0), (ol_thick - 1, actual_t_pad))
                draw_rectfill(b_img, (0, 0), (ol_thick - 1, actual_b_pad))

                # Draw right edge that appears in the top/bottom segments
                draw_rectfill(t_img, (w - ol_thick, 0), (w, actual_t_pad))
                draw_rectfill(b_img, (w - ol_thick, 0), (w, actual_b_pad))

            # Add labels to top/bottom segments if needed
            if self._label_t is not None and actual_t_pad > 0:
                t_xy_norm = self.style.top_label_xy_norm
                t_anchor = self.style.top_label_xy_norm
                t_offset = self.style.top_label_xy_norm
                t_margin = (0, 0)
                self.style.text_top.xy_norm(t_img, self._label_t, t_xy_norm, t_anchor, t_offset, t_margin)
            if self._label_b is not None and actual_b_pad > 0:
                b_xy_norm = self.style.bottom_label_xy_norm
                b_anchor = self.style.bottom_label_xy_norm
                b_offset = self.style.bottom_label_xy_norm
                b_margin = (0, 0)
                self.style.text_top.xy_norm(b_img, self._label_b, b_xy_norm, b_anchor, b_offset, b_margin)

            # Record images so that we don't need to re-generate every frame
            self._cache_t = t_img
            self._cache_b = b_img
            self._cache_l = l_img
            self._cache_r = r_img

            # Figure out size of new interaction region
            base_x1, base_y1 = self._cb_region.x1 + actual_l_pad, self._cb_region.y1 + actual_t_pad
            base_x2, base_y2 = base_x1 + base_w, base_y1 + base_h
            self._base_item._update_cb_region(base_x1, base_y1, base_x2, base_y2)

        out_img = np.hstack((self._cache_l, base_img, self._cache_r))
        return np.vstack((self._cache_t, out_img, self._cache_b))

    def _get_dynamic_aspect_ratio(self) -> SelfType:
        return self._base_item._get_dynamic_aspect_ratio()

    def _get_height_and_width_without_hint(self) -> HWPX:
        return self._base_item._get_height_and_width_without_hint()

    def _get_height_given_width(self, w: int) -> int:
        return self._base_item._get_height_given_width(w)

    def _get_width_given_height(self, h: int) -> int:
        return self._base_item._get_width_given_height(h)

    def _update_cb_region(self, x1, y1, x2, y2) -> SelfType:
        """
        Special override. This is needed to ensure the base item
        gets the proper region sizing/offset in case the parent
        (e.g. this class) has it's region updated after rendering,
        as caching can prevent direct updates to base item region sizing!
        """

        # Update own region
        self._cb_region.resize(x1, y1, x2, y2)

        # Tell base item to update region accordingly
        left_pad, top_pad = self._cache_l.shape[1], self._cache_t.shape[0]
        base_h, base_w = self._base_item._cb_region.h, self._base_item._cb_region.w
        base_x1, base_y1 = x1 + left_pad, y1 + top_pad
        base_x2, base_y2 = base_x1 + base_w, base_y1 + base_h
        self._base_item._update_cb_region(base_x1, base_y1, base_x2, base_y2)

        return self

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def _read_flex_values(
    item_is_flexible_list: list[bool],
    flex: tuple[float] | None,
    allow_undersizing: bool = True,
) -> tuple[float]:
    """
    Helper used to compute normalized flex values
    - 'None' values are interpreted as 'fallback' to item flexibility
    - If allow_undersizing is False, then normalized flex values will sum to 1
    - If allow_undersizing is True, then values less than 1 will remain as-is
      (so sum can be less than 1). This allows for 'shrinking' UI elements
      below given space allocation. For example, flex=(0.25, 0.25), would
      result in the items taking up only half of the total available space.
    """

    # If no flex sizing is given, default to 'fallback' for every item
    if flex is None:
        flex = [None] * len(item_is_flexible_list)

    # Sanity check, make sure we have flex sizing for each callback item
    flex = tuple(flex)
    num_items = len(item_is_flexible_list)
    assert len(flex) == num_items, f"Flex error! Must match number of entries ({num_items}), got: flex={flex}"

    # Iterpret flex values of None as fallback to item flexibility
    out_flex = (val if val is not None else float(is_flex) for val, is_flex in zip(flex, item_is_flexible_list))
    out_flex = tuple(max(0, val) for val in out_flex)

    # Normalize flex values so they can be used as weights when deciding render sizes
    total_flex = sum(out_flex)
    if allow_undersizing and total_flex < 0.99:
        total_flex = 1
    elif total_flex <= 0:
        total_flex = 1
    return tuple(float(val) / float(total_flex) for val in out_flex)
