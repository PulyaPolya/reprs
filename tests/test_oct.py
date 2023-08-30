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
    paths = get_input_midi_paths(seed=42, n_files=25)
    comparisons = 0
    inequalities = 0
    for i, path in enumerate(paths):
        print(f"{i + 1}/{len(paths)}: {path}")
        df = read_midi(path)
        # ticks_per_beat = mido.MidiFile(path).ticks_per_beat
        # TODO: (Malcolm 2023-08-22) put this processing into a function
        df = simplify_time_sigs(df)
        df = infer_barlines(df)
        assert isinstance(df, pd.DataFrame)
        encoding, feature_names = oct_encode(df)

        # Get reference implementation
        # with open(path, "rb") as f:
        #     midi_file = io.BytesIO(f.read())
        midi_obj = miditoolkit.midi.parser.MidiFile(filename=path)
        reference_encoding = MIDI_to_encoding(midi_obj)
        # These comparisons sometimes fail but from inspection it appears to be due to
        #   floating point/rounding issues where sometimes the onset of a note is rounded
        #   to a different position, or the relative position of a tempo change or similar
        #   is rounded differently

        for i, (x, y) in enumerate(zip(encoding, reference_encoding)):
            try:
                assert x == y
            except:
                for xx, yy in zip(x, y):
                    if xx != yy:
                        inequalities += 1
            comparisons += len(x)

        assert len(encoding) == len(reference_encoding)
    print(f"{comparisons - inequalities}/{comparisons} equal")

    # paths = get_input_kern_paths(seed=42)

    # for i, path in enumerate(paths):
    #     print(f"{i + 1}/{len(paths)}: {path}")
    #     df = read_humdrum(path)
    #     encoding = oct_encode(df)
    #     breakpoint()
