import io
import os
import random
import subprocess
from functools import partial

import pandas as pd
from music_df import read_krn

HUMDRUM_DATA_PATH = os.getenv("HUMDRUM_DATA")
MIDIDATA_PATH = os.getenv("MIDIDATA_PATH")

read_humdrum = partial(read_krn, sort=True)


def get_input_kern_paths(seed=None):
    assert HUMDRUM_DATA_PATH is not None
    krn_paths = [
        os.path.join(dirpath, filename)
        for (dirpath, dirs, files) in os.walk(HUMDRUM_DATA_PATH)
        for filename in (dirs + files)
        if filename.endswith(".krn")
    ]
    n_files = os.getenv("N_KERN_FILES")
    if n_files is not None:
        if seed is not None:
            random.seed(seed)
        krn_paths = random.sample(krn_paths, k=int(n_files))
    return krn_paths


def get_input_midi_paths(seed=None, n_files=1):
    assert MIDIDATA_PATH is not None
    midi_paths = [
        os.path.join(dirpath, filename)
        for (dirpath, dirs, files) in os.walk(MIDIDATA_PATH)
        for filename in (dirs + files)
        if filename.endswith(".mid")
    ]
    if n_files is not None:
        if seed is not None:
            random.seed(seed)
        midi_paths = random.sample(midi_paths, k=int(n_files))
    return midi_paths


def has_unison(df, note_i):
    n = df.loc[note_i]
    df[(df.pitch == n.pitch) & (df.release >= n.onset) & (df.onset <= n.release)]
    return len(n) > 1
