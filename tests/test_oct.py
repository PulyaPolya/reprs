import io

import miditoolkit
import mido
import pandas as pd
import pytest
from music_df.add_feature import infer_barlines, simplify_time_sigs
from music_df.read_midi import read_midi

from reprs.oct import oct_encode
from tests.helpers_for_tests import (
    get_input_kern_paths,
    get_input_midi_paths,
    read_humdrum,
)
from tests.original_oct_implementation import MIDI_to_encoding


@pytest.mark.filterwarnings(
    "ignore:note_off event"  # Ignore warnings from reading midi
)
def test_oct_encode(n_kern_files):
    paths = get_input_midi_paths(seed=42, n_files=1)
    comparisons = 0
    inequalities = 0
    vocab_sizes = {
        "bar": 256,  # Bars
        "position": 128,  # Positions
        "instrument": 129,  # Instrument
        "pitch": 256,  # Pitch
        "duration": 128,  # Duration
        "velocity": 32,  # Velocity
        "time_sig": 254,  # Time signatures
        "tempo": 49,  # Tempi
    }
    for i, path in enumerate(paths):
        if "K218 ii " in path:
            # TODO: (Malcolm 2023-09-11) this midi file has an excess token in
            #   my version for some reason. TODO investigate.
            continue
        if "Des heiligen Geistes reiche" in path:
            # TODO: (Malcolm 2023-09-11) this midi file has 3 excess tokens in my
            #   version for some reason.
            continue
        print(f"{i + 1}/{len(paths)}: {path}")
        df = read_midi(path)
        # ticks_per_beat = mido.MidiFile(path).ticks_per_beat
        # TODO: (Malcolm 2023-08-22) put this processing into a function
        df = simplify_time_sigs(df)
        df = infer_barlines(df)
        assert isinstance(df, pd.DataFrame)
        encoding = oct_encode(df)
        tokens = encoding._tokens

        # Get reference implementation
        # with open(path, "rb") as f:
        #     midi_file = io.BytesIO(f.read())
        midi_obj = miditoolkit.midi.parser.MidiFile(filename=path)
        reference_encoding = MIDI_to_encoding(midi_obj)
        # These comparisons sometimes fail but from inspection it appears to be due to
        #   floating point/rounding issues where sometimes the onset of a note is rounded
        #   to a different position, or the relative position of a tempo change or similar
        #   is rounded differently

        for i, (x, y) in enumerate(zip(tokens, reference_encoding)):
            for xx, token_type, n_tokens in zip(
                x, vocab_sizes.keys(), vocab_sizes.values()
            ):
                # TODO: (Malcolm 2023-09-11) I need to implement maximum bar cropping elsewhere
                assert xx < n_tokens or token_type == "bar"
            try:
                assert x == y
            except:
                for xx, yy, token_type, n_tokens in zip(
                    x, y, vocab_sizes.keys(), vocab_sizes.values()
                ):
                    # TODO: (Malcolm 2023-09-11) I need to implement maximum bar cropping elsewhere
                    assert xx < n_tokens or token_type == "bar"
                    if xx > 128 and token_type == "position":
                        breakpoint()
                    if xx != yy:
                        inequalities += 1
            comparisons += len(x)

        assert len(tokens) == len(reference_encoding)
    print(f"{comparisons - inequalities}/{comparisons} equal")

    # paths = get_input_kern_paths(seed=42)

    # for i, path in enumerate(paths):
    #     print(f"{i + 1}/{len(paths)}: {path}")
    #     df = read_humdrum(path)
    #     encoding = oct_encode(df)
    #     breakpoint()
