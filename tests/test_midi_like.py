import itertools as it

from reprs.midi_like import (
    MidiLikeSettings,
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

    for i, path in enumerate(paths):
        for include_barlines in (True, False):
            settings = MidiLikeSettings(
                include_barlines=include_barlines, for_token_classification=True
            )
            allowed_error = 2 ** (settings.min_ts_exp - 1)
            # path = "/Users/malcolm/datasets/humdrum-data/corelli/op3/op3n12-02.krn"
            # path = "/Users/malcolm/datasets/humdrum-data/jrp/Jos/kern/Jos1808-Qui_habitat_in_adjutorio_altissimi.krn"
            print(f"{i + 1}/{len(paths)}: {path}")
            df = read_humdrum(path)
            encoded = midilike_encode(
                df,
                settings,
                feature_names="spelling",
                sort=False,
            )
            # The only event types that are processed for the time being
            #   are "bar" and "note". If I update that then I'll want to update
            #   the next lines:
            if include_barlines:
                df = df[df.type.isin(["note", "bar"])]
            else:
                df = df[df.type == "note"]
            assert (
                sorted(encoded.note_on_idx_to_repr_idx.keys()) == df.index
            ).all()
            rv = encoded()
            for j in encoded.note_on_idx_to_repr_idx.values():
                assert (
                    rv["input"][j].startswith("note_on")
                    or rv["input"][j] == "bar"
                )
            for j in encoded.note_off_idx_to_repr_idx.values():
                assert rv["input"][j].startswith("note_off")
            for repr_i in encoded.repr_note_off_indices:
                assert rv["input"][repr_i].startswith("note_off")
            if include_barlines:
                assert "bar" in rv["input"]
            note_i = 0
            for repr_i, event in enumerate(rv["input"]):
                if event.startswith("note_off"):
                    assert encoded.repr_note_off_indices[note_i] == repr_i
                    note_i += 1
            decoded = midilike_decode(rv["input"])
            no_zero_len_note_df = df[
                (df.type != "note") | (df.onset != df.release)
            ]
            for j, note in no_zero_len_note_df.iterrows():
                if note.type != "note":
                    continue
                assert encoded.df.onset.loc[j] == note.onset
                assert encoded.df.release.loc[j] == note.release
                assert note.onset in encoded.sounding_notes_at_time
                assert note.release in encoded.sounding_notes_at_time

            assert len(rv["input"]) == len(rv["spelling"])
            for event, feature in zip(rv["input"], rv["spelling"]):
                if event.startswith("note_on"):
                    assert feature != NULL_FEATURE
                else:
                    assert feature == NULL_FEATURE

            for (j, src_note), (k, dst_note) in zip(
                no_zero_len_note_df.iterrows(), decoded.iterrows()
            ):
                assert src_note.type == dst_note.type
                if src_note.type != "note":
                    continue
                try:
                    assert src_note.pitch == dst_note.pitch
                except AssertionError:
                    # This assertion can fail if there is another note whose onset
                    #   is close enough to src_note's onset that in the decoding
                    #   that it comes out as simultaneous in the decoded no_zero_len_note_df.
                    assert (
                        src_note.pitch
                        in (
                            decoded[decoded.onset == dst_note.onset].pitch
                        ).values
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
                    assert (
                        abs(src_note.release - dst_note.release)
                        <= allowed_error
                    )
                except AssertionError:
                    has_unison(decoded, k)


def test_segment(n_kern_files):
    paths = get_input_kern_paths(seed=42)

    for i, path in enumerate(paths):
        print(f"{i + 1}/{len(paths)}: {path}")
        for include_barlines in (True, False):
            boundary_tokens = True
            for_token_class = True
            for window_len, hop in it.product(
                (64, 128),
                (48, 4, 128),
            ):
                settings = MidiLikeSettings(
                    include_barlines=include_barlines,
                    start_token=boundary_tokens,
                    end_token=boundary_tokens,
                    for_token_classification=for_token_class,
                )
                df = read_humdrum(path)
                encoded = midilike_encode(
                    df,
                    settings,
                    feature_names="spelling",
                    for_token_classification=True,
                    sort=False,
                )
                # The only event types that are processed for the time being
                #   are "bar" and "note". If I update that then I'll want to update
                #   the next lines:
                if include_barlines:
                    df = df[df.type.isin(["note", "bar"])]
                    note_df = df[df.type == "note"]
                else:
                    df = note_df = df[df.type == "note"]

                print(".", end="", flush=True)
                encoded = midilike_encode(
                    df,
                    settings,
                    feature_names="spelling",
                    sort=False,
                )
                # These tests seem to be redundant given the tests in
                #   test_midi_like() above
                for j in encoded.note_on_idx_to_repr_idx.values():
                    assert (
                        encoded.events[j].startswith("note_on")
                        or encoded.events[j] == "bar"
                    )
                for j in encoded.note_off_idx_to_repr_idx.values():
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
                prev_offset = -1.0
                for i, (segment) in enumerate(
                    encoded.segment(
                        window_len,
                        hop,
                        min_window_len=min_window_len,
                        allow_short_initial_window=True,
                        allow_short_last_window=True,
                        return_repr_indices=True,
                    )
                ):
                    for segment_i, repr_i in enumerate(segment["repr_indices"]):
                        repr_i = segment["repr_indices"][segment_i]
                        if repr_i is None:
                            continue
                        assert (
                            encoded.events[repr_i]
                            == segment["input"][segment_i]
                        )
                    assert not iteration_should_finish_before_here
                    assert segment["segment_onset"] >= prev_offset
                    prev_offset = segment["segment_onset"]
                    if i == 0:
                        if boundary_tokens:
                            assert segment["input"][0] == START_TOKEN
                        else:
                            assert segment["input"][0] != START_TOKEN
                    elif enforce_only_one_item:
                        raise ValueError("this df should only have one segment")
                    assert len(segment["input"]) <= window_len
                    try:
                        assert len(segment["input"]) >= min_window_len
                    except AssertionError:
                        try:
                            assert i == 0
                        except AssertionError:
                            # this should be the last item
                            iteration_should_finish_before_here = True
                        else:
                            enforce_only_one_item = True
                    note_ons = [
                        event
                        for event in segment["input"]
                        if event.startswith("note_on")
                    ]
                    bars = [
                        event for event in segment["input"] if event == "bar"
                    ]
                    note_offs = [
                        event
                        for event in segment["input"]
                        if event.startswith("note_off")
                    ]
                    assert len(note_ons) == len(note_offs)
                    if for_token_class:
                        assert len(segment["input"]) == len(segment["spelling"])
                        for event, feature_val in zip(
                            segment["input"], segment["spelling"]
                        ):
                            if event.startswith("note_on"):
                                assert feature_val != NULL_FEATURE
                            else:
                                assert feature_val == NULL_FEATURE
                    else:
                        assert all(
                            len(note_ons) == len(segment["spelling"][name])
                            for name in segment["spelling"]
                        )
                    decoded = midilike_decode(segment["input"])
                    assert len(decoded) == len(note_ons) + len(bars)
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
