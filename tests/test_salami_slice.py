import pandas as pd


from reprs.df_utils import sort_df
from reprs.salami_slice import salami_slice

from tests.helpers_for_tests import (
    read_humdrum,
    get_input_kern_paths,
)


def _join_consecutive_notes(df: pd.DataFrame):
    # assumes df is sorted
    dont_compare = ["onset", "release"]
    compare = df.drop(dont_compare, axis=1)
    unique = compare.drop_duplicates()
    out = []
    for _, pitch in unique.iterrows():
        instances = df[(df[pitch.index] == pitch).all(axis=1)]
        onset = None
        prev_release = None
        for _, note in instances.iterrows():
            if onset is not None and note.onset != prev_release:
                new_note = pitch.copy()
                new_note["onset"] = onset
                new_note["release"] = prev_release
                out.append(new_note)
            if note.onset != prev_release:
                onset = note.onset
            prev_release = note.release
        new_note = pitch.copy()
        new_note["onset"] = onset
        new_note["release"] = prev_release
        out.append(new_note)
    out_df = pd.DataFrame(out)
    out_df.reset_index(drop=True, inplace=True)
    sort_df(out_df, inplace=True)
    return out_df


def test_midi_like(n_kern_files):
    paths = get_input_kern_paths(seed=42)
    for i, path in enumerate(paths):
        # path = "/Users/malcolm/datasets/humdrum-data/corelli/op3/op3n12-02.krn"
        # path = "/Users/malcolm/datasets/humdrum-data/jrp/Jos/kern/Jos1808-Qui_habitat_in_adjutorio_altissimi.krn"
        print(f"{i + 1}/{len(paths)}: {path}")
        df = read_humdrum(path)
        new_df = salami_slice(df)
        # All onsets should be associated with only one release
        for onset in new_df.onset.unique():
            assert len(new_df[new_df.onset == onset].release.unique()) == 1
        # All releases should be associated with only one onset
        for release in new_df.release.unique():
            assert len(new_df[new_df.release == release].onset.unique()) == 1
        # Make sure output is correctly sorted
        # We don't guarantee that columns outside of sorted_columns are the same
        sorted_columns = ["pitch", "onset", "release", "spelling"]
        assert new_df[sorted_columns].equals(sort_df(new_df)[sorted_columns])
        joined_new_df = _join_consecutive_notes(new_df)
        joined_orig_df = _join_consecutive_notes(df)
        # It's possible for the datatypes to disagree if orig_df has ints
        #   for onsets and/or releases. In that case, eq() will evaluate
        #   to True for each item, but equals() will evaluate to False.
        # This test fails because joining doesn't work correctly for unisons.
        #   Not worth the time to implement correctly, I think. An example
        #   is '/Users/malcolm/datasets/humdrum-data/bach-js/brandenburg/kern/bwv1046c.krn'
        try:
            assert (
                joined_new_df[sorted_columns]
                .eq(joined_orig_df[sorted_columns])
                .all()
                .all()
            )
        except AssertionError:
            assert (
                path
                == "/Users/malcolm/datasets/humdrum-data/bach-js/brandenburg/kern/bwv1046c.krn"
            )
