#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from time import perf_counter


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class TickRateLimiter:
    """Simple update limiter"""

    def __init__(self, num_ticks_per_update=1, update_on_first_call=True):
        self._num_ticks_per_update = num_ticks_per_update
        self._curr_ticks = num_ticks_per_update if update_on_first_call else 0

    def tick(self):
        self._curr_ticks += 1
        need_update = self._curr_ticks >= self._num_ticks_per_update
        if need_update:
            self._curr_ticks = 0
        return need_update

    def set_rate(self, num_ticks_per_update: int | float):
        self._num_ticks_per_update = num_ticks_per_update
        return self

    def set_rate_lerp(self, t0_value: float, t1_value: float, t: float):
        """
        Helper for setting the tick count based on a variable control
        Sets count according to:
            t0_value * (1-t) + t1_value * t
        Note: There is no clamping on the weighting (t) value!
        """
        self._num_ticks_per_update = t0_value * (1.0 - t) + t1_value * t
        return self


class FPSLimiter:
    """Simple FPS update limiter"""

    def __init__(self, target_frames_per_second=60.0, update_on_first_call=True):
        self._sec_per_frame = 1.0 / target_frames_per_second
        self._next_update_sec = -1 if update_on_first_call else perf_counter() + self._sec_per_frame
        self._last_tick_sec = perf_counter()

    def tick(self):

        curr_time_sec = perf_counter()
        need_update = curr_time_sec >= self._next_update_sec
        if need_update:
            num_updates = 1 + (curr_time_sec - self._next_update_sec) // self._sec_per_frame
            self._next_update_sec += num_updates * self._sec_per_frame
            
        # Keep track of how much time has passed
        delta_t_sec = curr_time_sec - self._last_tick_sec
        self._last_tick_sec = curr_time_sec

        return need_update, delta_t_sec

    def set_rate(self, frames_per_second: float):
        self._sec_per_frame = 1.0 / max(1, frames_per_second)

    def set_rate_lerp(self, t0_value: float, t1_value: float, t: float):
        """
        Helper for setting the frame rate based on a variable control
        using linear interpolation between two value.
        Sets fps according to:
            t0_value * (1-t) + t1_value * t
        Note: There is no clamping on the weighting (t) value!
        """

        target_fps = t0_value * (1.0 - t) + t1_value * t
        self._sec_per_frame = 1.0 / max(1, target_fps)
        return self

    def __iter__(self):
        return self

    # .................................................................................................................

    def __next__(self) -> bool:
        """
        Iterator that 'ticks' the FPS counter.
        Returns: need_update
        (when appropriate amount of time has passed, based on target frame rate)
        """

        return self.tick()


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions
