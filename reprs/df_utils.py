import typing as t
import numpy as np
import pandas as pd

from reprs.utils import get_item_less_than_or_eq


def get_eligible_onsets(
    df: pd.DataFrame, keep_onsets_together: bool = True
) -> t.Union[pd.Index, np.array]:
    """
    >>> df = pd.DataFrame({
    ...     "pitch": [60, 64, 60, 64, 60, 64, 60, 64],
    ...     "onset": [0, 0, 1, 1, 1.5, 2.0, 3.0, 3.0],
    ...     "release": [1, 1, 1.5, 2.0, 3.0, 3.0, 4.0, 4.5],
    ... })
    >>> df
       pitch  onset  release
    0     60    0.0      1.0
    1     64    0.0      1.0
    2     60    1.0      1.5
    3     64    1.0      2.0
    4     60    1.5      3.0
    5     64    2.0      3.0
    6     60    3.0      4.0
    7     64    3.0      4.5
    >>> get_eligible_onsets(df)
    array([0, 2, 4, 5, 6])
    """
    if not keep_onsets_together:
        return df.index
    onset_indices = np.unique(df.onset, return_index=True)[1]
    return onset_indices


def get_eligible_releases(
    df: pd.DataFrame, keep_releases_together: bool = True
) -> pd.Series:
    """

    Returns a series where the Index gives the indices into the dataframe
    and the values are the associated release times.
    >>> df = pd.DataFrame({
    ...     "pitch": [60, 64, 60, 64, 60, 64, 60, 64],
    ...     "onset": [0, 0, 1, 1, 1.5, 2.0, 3.0, 3.0],
    ...     "release": [1, 1, 1.5, 2.0, 3.0, 3.0, 4.0, 4.5],
    ... })
    >>> df
       pitch  onset  release
    0     60    0.0      1.0
    1     64    0.0      1.0
    2     60    1.0      1.5
    3     64    1.0      2.0
    4     60    1.5      3.0
    5     64    2.0      3.0
    6     60    3.0      4.0
    7     64    3.0      4.5
    >>> get_eligible_releases(df)
    1    1.0
    2    1.5
    3    2.0
    5    3.0
    6    4.0
    7    4.5
    Name: release, dtype: float64
    """
    if not keep_releases_together:
        return df.release
    df2 = df.sort_values(
        by="pitch", inplace=False, ignore_index=False, kind="mergesort"
    )
    df2 = df2.sort_values(
        by="release", inplace=False, ignore_index=False, kind="mergesort"
    )
    release_indices = (len(df2) - 1) - np.unique(
        np.flip(df2.release.values), return_index=True
    )[1]
    out = df2.iloc[release_indices]["release"]
    return out


def get_df_segment_indices(eligible_onsets, eligible_releases, target_len):
    """
    # >>> eligible_onsets = list(range(32))
    # >>> eligible_releases = list(range(32))
    # >>> list(get_df_segment_indices(eligible_onsets, eligible_releases, 8))
    # [(0, 8), (8, 16), (16, 24), (24, 32)]

    # >>> eligible_onsets = [i * 2 for i in range(16)]
    # >>> eligible_releases = [i * 2 + 1 for i in range(16)]
    # >>> list(get_df_segment_indices(eligible_onsets, eligible_releases, 8))
    # [(0, 8), (8, 16), (16, 24), (24, 32)]

    # >>> eligible_onsets = [0, 3, 7, 14]
    # >>> eligible_releases = [2, 3, 6, 12, 13, 17]
    # >>> list(get_df_segment_indices(eligible_onsets, eligible_releases, 8))
    # [(0, 7), (7, 14), (14, 18)]

    We aim for target_len, but there is no firm limit on how long a segment
    might be. We depend on eligible_onsets/eligible_releases to be fairly
    evenly distributed to avoid segments that are far too long (or short).
    >>> eligible_onsets = [0, 1, 14, 15]
    >>> eligible_releases = [2, 3, 17]
    >>> list(get_df_segment_indices(eligible_onsets, eligible_releases, 8))
    [(0, 4), (1, 18)]

    >>> eligible_onsets = [0, 1, 14, 15]
    >>> eligible_releases = [16, 17]
    >>> list(get_df_segment_indices(eligible_onsets, eligible_releases, 8))
    [(0, 17), (15, 18)]

    Releases before the first eligible onset are ignored.
    >>> eligible_onsets = [14, 15]
    >>> eligible_releases = [0, 1, 16, 17]
    >>> list(get_df_segment_indices(eligible_onsets, eligible_releases, 8))
    [(14, 18)]

    There shouldn't be any other circumstance in which onsets or releases are
    skipped.
    """
    # assumes df has a range index
    start_i = None
    end_i = eligible_releases[0] - 1
    max_release_i = eligible_releases[-1]
    while end_i < max_release_i:
        if start_i is None:
            start_i = eligible_onsets[0]
        else:
            try:
                start_i = get_item_less_than_or_eq(
                    eligible_onsets, end_i + 1, min_val=start_i + 1
                )
            except ValueError:  # pylint: disable=try-except-raise
                # We should never get here, I think this is a bug if we do
                raise
        # we calculate end_i *inclusively*, then add 1 to it to return
        #   an exclusive boundary appropriate for slicing in Python
        end_i = get_item_less_than_or_eq(
            eligible_releases,
            # We need to subtract 1 from target_len because we are
            #   calculating an inclusive boundary
            start_i + target_len - 1,
            min_val=max(start_i + 1, end_i + 1),
        )
        yield start_i, end_i + 1


def segment_df(df, target_len):
    eligible_onsets = get_eligible_onsets(df)
    eligible_releases = get_eligible_releases(df).index
    for start_i, end_i in get_df_segment_indices(
        eligible_onsets, eligible_releases, target_len
    ):
        yield df[start_i:end_i]


def sort_df(df, inplace=False):
    if not inplace:
        df = df.sort_values(
            by="release",
            axis=0,
            inplace=False,
            ignore_index=True,
            key=lambda x: 0 if x is None else x,
        )
    else:
        df.sort_values(
            by="release",
            axis=0,
            inplace=True,
            ignore_index=True,
            key=lambda x: 0 if x is None else x,
        )
    df.sort_values(
        by="pitch",
        axis=0,
        inplace=True,
        ignore_index=True,
        key=lambda x: 128 if x is None else x,
        kind="mergesort",  # default sort is not stable
    )
    df.sort_values(
        by="onset",
        axis=0,
        inplace=True,
        ignore_index=True,
        kind="mergesort",  # default sort is not stable
    )
    return df
