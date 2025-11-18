#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from collections import OrderedDict, deque

# For type hints
from typing import Any


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class ToggleStateTracker:
    """Helper used to keep track of a toggle (boolean) variable"""

    def __init__(self, initial_state: bool = False, start_as_changed=True):
        self._is_changed = start_as_changed
        self._curr_state = initial_state
        return

    def read(self) -> tuple[bool, bool]:
        """Returns: is_changed, current_state"""
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed, self._curr_state

    def toggle(self, new_state: bool | None = None) -> bool:
        """Flip the current state (or set to a specific state if given a boolean input)"""

        old_state = self._curr_state
        self._curr_state = not self._curr_state if new_state is None else new_state
        if old_state != self._curr_state:
            self._is_changed = True

        return self._curr_state

    def set_is_changed(self, is_changed: bool = True):
        """
        Allows for artificially modifying the 'is changed' state.
        Useful for forcing a read update or for suppressing updates.
        """
        self._is_changed = is_changed
        return self

    # .................................................................................................................


class ValueChangeTracker:
    """
    Helper used to keep track of changes to a value
    Meant for more efficient use of variables, for example
    only triggering events on value changes, rather than
    just the state value.
    """

    # .................................................................................................................

    def __init__(self, initial_value: Any = None):
        self._prev = initial_value

    def is_changed(self, new_value: Any) -> bool:
        """Check if new_value is no equal to the value when last checked"""
        is_changed, self._prev = (self._prev != new_value), new_value
        return is_changed

    def __call__(self, new_value: Any) -> bool:
        """Same as .is_changed. Provided for readability in code"""
        is_changed, self._prev = (self._prev != new_value), new_value
        return is_changed

    # .................................................................................................................


class DelayableValueChangeTracker:
    """
    Similar to 'ValueChangeTracker'.
    Used to keep track of changes to a value, but with
    the potential to 'delay' the record keeping of the new value.
    Example usage:

        # Set up tracker
        data_size = 10
        change_tracker = DelayableValueChangeTracker(data_size)

        # ... some other code ...

        # Check if value changes, but only if given a non-None value
        new_data_size = None
        is_changed = change_tracker(new_data_size, record_value=new_data_size is not None)
    """

    # .................................................................................................................

    def __init__(self, initial_value: Any = None):
        self._prev_value = initial_value

    def __call__(self, value: Any, record_value: bool = False) -> bool:
        """Equivalent to calling .is_changed(...)"""
        return self.is_changed(value, record_value)

    def is_changed(self, value: Any, record_value: bool = False) -> bool:
        """Function used to check if the given value is different from the previously recorded value"""
        value_changed = value != self._prev_value
        if value_changed and record_value:
            self._prev_value = value
        return value_changed

    def record(self, value: Any):
        """Records the given value for use in future 'did it change' checks"""
        self._prev_value = value
        return self

    def clear(self, clear_value: Any = None):
        """Helper that is identical to 'record(None)' but may be nicer to use to indicate intent"""
        self._prev_value = clear_value

    # .................................................................................................................


class MaxLengthKVStorage:
    """
    Helper object used to implement key-value storage (e.g. a dictionary)
    with a max-length. If items are added beyond the max length, the oldest
    items will be eliminated (i.e. FIFO).
    """

    # .................................................................................................................

    def __init__(self, max_length: int = 30):
        self.data = OrderedDict()
        self._max_length = max_length

    def __repr__(self):
        return self.data.__repr__()

    def __len__(self):
        return len(self.data.keys())

    def __iter__(self):
        return self.data.__iter__()

    def __setitem__(self, key, value):
        return self.store(key, value)

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    # .................................................................................................................

    def items(self):
        return self.data.items()

    def clear(self):
        self.data.clear()

    def store(self, key: Any, value: Any):
        """
        Store new key-value pair.
        Older entries will be removed if the storage size
        exceeds the set maximum length.
        """
        self.data[key] = value
        if len(self.data) > self._max_length:
            self.data.popitem(last=False)
        return self

    def get(self, key: Any, value_if_missing: Any = None) -> Any:
        """Retrieve item from storage"""
        return self.data.get(key, value_if_missing)

    # .................................................................................................................


class UndoRedoList:
    """
    Class used to implement a basic list-like storage object
    that is capable of undo/redo/clear operations.
    """

    # Flags used to store which actions have been taken
    _ACT_APPEND = 0
    _ACT_CLEAR = 1

    # .................................................................................................................

    def __init__(self, max_undos=50, allow_undo_clear=True):
        self._data_list = []
        self._forward_actions = deque([], maxlen=max_undos)
        self._backward_actions = []
        self._allow_undo_clear = allow_undo_clear

    def __repr__(self):
        name = self.__class__.__name__
        num_data = len(self._data_list)
        num_undos = len(self._backward_actions)
        return f"{name} ({num_data} data, {num_undos} undos)"

    def __len__(self):
        return len(self._data_list)

    def __iter__(self):
        return iter(self._data_list)

    def __getitem__(self, index: int):
        return self._data_list[index]

    # .................................................................................................................

    def append(self, data):
        """Store new data entry"""
        self._forward_actions.append((self._ACT_APPEND, None))
        self._data_list.append(data)
        self._backward_actions = []
        return self

    def clear(self) -> bool:
        """Clear current data. Returns: had_data_to_clear"""
        have_data_to_clear = len(self._data_list) > 0
        if have_data_to_clear:
            if self._allow_undo_clear:
                self._forward_actions.append((self._ACT_CLEAR, self._data_list))
            else:
                self._forward_actions.clear()
            self._data_list = []
            self._backward_actions = []
        return have_data_to_clear

    # .................................................................................................................

    def undo(self) -> bool:
        """Reverse last action (e.g. append or clear). Returns: had_action_to_undo"""
        have_action_to_undo = len(self._forward_actions) > 0
        if have_action_to_undo:
            command, data = self._forward_actions.pop()
            if command == self._ACT_APPEND:
                assert data is None, f"Error with undo-append. Expecting no data, got: {data}"
                data = self._data_list.pop()
            elif command == self._ACT_CLEAR:
                self._data_list = data
                data = None
            self._backward_actions.append((command, data))
        return have_action_to_undo

    def redo(self) -> bool:
        """Redo last undo. Returns: had_action_to_redo"""
        have_action_to_redo = len(self._backward_actions) > 0
        if have_action_to_redo:
            command, data = self._backward_actions.pop()
            if command == self._ACT_APPEND:
                self._data_list.append(data)
                data = None
            elif command == self._ACT_CLEAR:
                assert data is None, f"Error with redo-clear. Expecting no data, got: {data}"
                data = self._data_list
                self._data_list = []
            self._forward_actions.append((command, data))
        return have_action_to_redo

    def reset(self):
        """Fully resets data (i.e. clear without the ability to undo)"""
        self._data_list = []
        self._forward_actions.clear()
        self._backward_actions = []
        return self

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions
