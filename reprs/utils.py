import numpy as np


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


def get_item_less_than_or_eq(sorted_array, val, min_val=None):
    """
    >>> get_item_less_than_or_eq([1, 3, 6], 3)
    3
    >>> get_item_less_than_or_eq([1, 3, 6], 4)
    3
    >>> get_item_less_than_or_eq([1, 3, 6], 4, min_val=2)
    3
    >>> get_item_less_than_or_eq([1, 3, 6], 4, min_val=4)
    6

    If value is < first item or if min is > last item, a ValueError will
    be raised:
    >>> get_item_less_than_or_eq([1, 3, 6], 0)
    Traceback (most recent call last):
    ValueError: 0 < first item of `sorted_array` 1
    >>> get_item_less_than_or_eq([1, 3, 6], 4, min_val=8)
    Traceback (most recent call last):
    ValueError: min_val=8 is greater than last item of `sorted_array` 6
    """
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
    out = sorted_array[idx]
    if min_val is None or out >= min_val:
        return out
    idx = get_idx_to_item_geq(sorted_array, min_val)
    return sorted_array[idx]
