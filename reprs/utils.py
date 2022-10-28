import numpy as np
import pandas as pd


def get_idx_to_item_leq(sorted_array, val):
    """
    >>> get_idx_to_item_leq([1, 3, 6], 3)
    1
    >>> get_idx_to_item_leq([1, 3, 6], 5)
    1
    >>> get_idx_to_item_leq([1, 3, 6], 10)
    2

    If values is < the first item in the array, will raise a ValueError.
    >>> get_idx_to_item_leq([1, 3, 6], 0)
    Traceback (most recent call last):
    ValueError: 0 < first item of `sorted_array` 1

    """
    if val < sorted_array[0]:
        raise ValueError(
            f"{val} < first item of `sorted_array` {sorted_array[0]}"
        )
    return np.searchsorted(sorted_array, val, "right") - 1


def get_idx_to_item_geq(sorted_array, val):
    """
    >>> get_idx_to_item_geq([1, 3, 6], 0)
    0
    >>> get_idx_to_item_geq([1, 3, 6], 1)
    0
    >>> get_idx_to_item_geq([1, 3, 6], 5)
    2

    If values is > the last item in the array, will raise a ValueError.
    >>> get_idx_to_item_geq([1, 3, 6], 7)
    Traceback (most recent call last):
    ValueError: 7 > last item of `sorted_array` 6
    """
    if val > sorted_array[-1]:
        raise ValueError(
            f"{val} > last item of `sorted_array` {sorted_array[-1]}"
        )
    return np.searchsorted(sorted_array, val, "left")


def _get_item_leq_sub(sorted_array, val, min_val):
    """Used by both get_item_leq and get_index_to_item_leq."""
    if min_val is not None:
        val = max(min_val, val)
    if val < sorted_array[0]:
        # TODO handle min case
        raise ValueError(
            f"{val} < first item of `sorted_array` {sorted_array[0]}"
        )
    if min_val is not None and min_val > sorted_array[-1]:
        raise ValueError(
            f"min_val={min_val} is greater than last item of "
            f"`sorted_array` {sorted_array[-1]}"
        )
    idx = get_idx_to_item_leq(sorted_array, val)
    if min_val is None or sorted_array[idx] >= min_val:
        return idx
    idx = get_idx_to_item_geq(sorted_array, min_val)
    return idx


def get_item_leq(sorted_array, val, min_val=None):
    """
    >>> get_item_leq([1, 3, 6], 3)
    3
    >>> get_item_leq([1, 3, 6], 4)
    3
    >>> get_item_leq([1, 3, 6], 4, min_val=2)
    3
    >>> get_item_leq([1, 3, 6], 4, min_val=4)
    6

    If value is < first item or if min is > last item, a ValueError will
    be raised:
    >>> get_item_leq([1, 3, 6], 0)
    Traceback (most recent call last):
    ValueError: 0 < first item of `sorted_array` 1
    >>> get_item_leq([1, 3, 6], 4, min_val=8)
    Traceback (most recent call last):
    ValueError: min_val=8 is greater than last item of `sorted_array` 6
    """
    return sorted_array[_get_item_leq_sub(sorted_array, val, min_val)]


def get_index_to_item_leq(sorted_series, val, min_val=None):
    """Same as get_item_leq except it expects a Pandas
    series, compares `val` to the values in the series and returns
    the index to the relevant item.

    >>> series = pd.Series([1, 3, 6], index=["a", "b", "z"])
    >>> get_index_to_item_leq(series, 3)
    'b'
    >>> get_index_to_item_leq(series, 4)
    'b'
    >>> get_index_to_item_leq(series, 4, min_val=2)
    'b'
    >>> get_index_to_item_leq(series, 4, min_val=4)
    'z'

    If value is < first item or if min is > last item, a ValueError will
    be raised:
    >>> get_index_to_item_leq(series, 0)
    Traceback (most recent call last):
    ValueError: 0 < first item of `sorted_array` 1
    >>> get_index_to_item_leq(series, 4, min_val=8)
    Traceback (most recent call last):
    ValueError: min_val=8 is greater than last item of `sorted_array` 6
    """
    return sorted_series.index[
        _get_item_leq_sub(sorted_series.values, val, min_val)
    ]
