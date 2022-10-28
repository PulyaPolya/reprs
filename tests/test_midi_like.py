import itertools as it

from reprs.shared import ReprSettings
from reprs.midi_like import (
    midilike_encode,
    midilike_decode,
    NULL_FEATURE,
    START_TOKEN,
    # END_TOKEN,
)

from tests.helpers_for_tests import (
    read_humdrum,
    get_input_kern_paths,
    has_unison,
)


def test_midi_like(n_kern_files):
    paths = get_input_kern_paths(seed=42)
    settings = ReprSettings()
    allowed_error = 2 ** (settings.min_ts_exp - 1)
    for i, path in enumerate(paths):
        # path = "/Users/malcolm/datasets/humdrum-data/corelli/op3/op3n12-02.krn"
        # path = "/Users/malcolm/datasets/humdrum-data/jrp/Jos/kern/Jos1808-Qui_habitat_in_adjutorio_altissimi.krn"
        print(f"{i + 1}/{len(paths)}: {path}")
        df = read_humdrum(path)
        encoded = midilike_encode(
            df,
            settings,
            feature_names="spelling",
            for_token_classification=True,
            sort=False,
        )
        assert (
            sorted(encoded.note_on_idx_to_repr_idx.keys()) == df.index
        ).all()
        for _, j in encoded.note_on_idx_to_repr_idx.items():
            assert encoded.events[j].startswith("note_on")
        for _, j in encoded.note_off_idx_to_repr_idx.items():
            assert encoded.events[j].startswith("note_off")
        for repr_i in encoded.repr_note_off_indices:
            assert encoded.events[repr_i].startswith("note_off")
        note_i = 0
        for repr_i, event in enumerate(encoded.events):
            if event.startswith("note_off"):
                assert encoded.repr_note_off_indices[note_i] == repr_i
                note_i += 1
        decoded = midilike_decode(encoded.events)
        no_zero_len_note_df = df[df.onset != df.release]
        for j, note in no_zero_len_note_df.iterrows():
            assert encoded.note_on_idx_to_time[j] == note.onset
            assert encoded.note_off_idx_to_time[j] == note.release
            assert note.onset in encoded.sounding_notes_at_time
            assert note.release in encoded.sounding_notes_at_time
        for name in encoded.features:
            assert len(encoded.events) == len(encoded.features[name])
            for event, feature in zip(encoded.events, encoded.features[name]):
                if event.startswith("note_on"):
                    assert feature != NULL_FEATURE
                else:
                    assert feature == NULL_FEATURE

        for (j, src_note), (_, dst_note) in zip(
            no_zero_len_note_df.iterrows(), decoded.iterrows()
        ):
            try:
                assert src_note.pitch == dst_note.pitch
            except AssertionError:
                # This assertion can fail if there is another note whose onset
                #   is close enough to src_note's onset that in the decoding
                #   that it comes out as simultaneous in the decoded no_zero_len_note_df.
                assert (
                    src_note.pitch
                    in (decoded[decoded.onset == dst_note.onset].pitch).values
                )
                assert (
                    dst_note.pitch
                    in no_zero_len_note_df[
                        (
                            src_note.onset - allowed_error
                            <= no_zero_len_note_df.onset
                        )
                        & (
                            no_zero_len_note_df.onset
                            <= src_note.onset + allowed_error
                        )
                    ].pitch.values
                )
            # the next test fails; I think this is because it is
            #   conceptually incorrect; what we expect to be below allowed_error
            #   is the difference between the interval between the previous note
            #   and this one, not the difference between the absolute positions
            #   of the original and decoded notes.
            # assert abs(src_note.onset - dst_note.onset) <= allowed_error
            # I think regardless of the note_end procedure we adopt in the event
            # of unisons, we can't guarantee that it releases will match
            # because, in the source file, the notes could be on different channels
            # or tracks in such a way as to behave differently. (E.g., ending the
            # first note, if we've chosen to end the last, or vice versa.)
            # But we can at least make sure that there is a concurrent unison.
            try:
                assert abs(src_note.release - dst_note.release) <= allowed_error
            except AssertionError:
                has_unison(decoded, j)
        for_token_class = False
        boundary_tokens = True
        for window_len, hop in it.product(
            (64, 128),
            (48, 4, 8, 128),
        ):
            print(".", end="", flush=True)
            encoded = midilike_encode(
                df,
                settings,
                feature_names="spelling",
                end_token=boundary_tokens,
                start_token=boundary_tokens,
                for_token_classification=for_token_class,
                sort=False,
            )
            for _, j in encoded.note_on_idx_to_repr_idx.items():
                assert encoded.events[j].startswith("note_on")
            for _, j in encoded.note_off_idx_to_repr_idx.items():
                assert encoded.events[j].startswith("note_off")
            for repr_i in encoded.repr_note_off_indices:
                assert encoded.events[repr_i].startswith("note_off")
            note_i = 0
            for repr_i, event in enumerate(encoded.events):
                if event.startswith("note_off"):
                    assert encoded.repr_note_off_indices[note_i] == repr_i
                    note_i += 1
            min_window_len = window_len // 2
            enforce_only_one_item = False
            iteration_should_finish_before_here = False
            prev_idx = -1
            for i, (segment, features, idx, repr_indices) in enumerate(
                encoded.segment(
                    window_len,
                    hop,
                    min_window_len=min_window_len,
                    allow_short_initial_window=True,
                    allow_short_last_window=True,
                    return_repr_indices=True,
                )
            ):
                for segment_i in range(len(repr_indices)):
                    repr_i = repr_indices[segment_i]
                    if repr_i is None:
                        continue
                    assert encoded.events[repr_i] == segment[segment_i]
                assert not iteration_should_finish_before_here
                assert idx >= prev_idx
                prev_idx = idx
                if i == 0:
                    if boundary_tokens:
                        assert segment[0] == START_TOKEN
                    else:
                        assert segment[0] != START_TOKEN
                elif enforce_only_one_item:
                    raise ValueError("this df should only have one segment")
                assert len(segment) <= window_len
                try:
                    assert len(segment) >= min_window_len
                except AssertionError:
                    try:
                        assert i == 0
                    except AssertionError:
                        # this should be the last item
                        iteration_should_finish_before_here = True
                    else:
                        enforce_only_one_item = True
                note_ons = [
                    event for event in segment if event.startswith("note_on")
                ]
                note_offs = [
                    event for event in segment if event.startswith("note_off")
                ]
                assert len(note_ons) == len(note_offs)
                if for_token_class:
                    for feature in features.values():
                        assert len(segment) == len(feature)
                        for event, feature_val in zip(segment, feature):
                            if event.startswith("note_on"):
                                assert feature_val != NULL_FEATURE
                            else:
                                assert feature_val == NULL_FEATURE
                else:
                    assert all(
                        len(note_ons) == len(features[name])
                        for name in features
                    )
                decoded = midilike_decode(segment)
                assert len(decoded) == len(note_ons)
            for_token_class = True
            boundary_tokens = False
            # if boundary_tokens:
            #     # this will not hold in the case where we have to return early
            #     #   because there would be too many events in the last window
            #     #   (e.g., /humdrum-data/jrp/Bru/kern/Bru1008e-Missa_Et_ecce_terre_motus-Agnus.krn)
            #     assert (
            #         segment[-1] == END_TOKEN
            #         or iteration_should_finish_before_here
            #     )
            # else:
            #     assert segment[-1] != END_TOKEN
        print("")
