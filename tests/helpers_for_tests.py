import subprocess
import io
import os
import random

from functools import partial

import pandas as pd

from music_df import read_krn

TOTABLE = os.getenv("TOTABLE")
HUMDRUM_DATA_PATH = os.getenv("HUMDRUM_DATA")

read_humdrum = partial(read_krn, sort=True)

# def read_humdrum(path, remove_zero_length_notes=True):
#     result = subprocess.run(
#         [TOTABLE, path], check=True, capture_output=True
#     ).stdout.decode()
#     df = pd.read_csv(io.StringIO(result), sep="\t")
#     df.attrs["score_name"] = path
#     if remove_zero_length_notes:
#         df = df[(df.type != "note") | (df.release > df.onset)].reset_index(
#             drop=True
#         )
#     df = sort_df(df, inplace=True)
#     return df


def get_input_kern_paths(seed=None):
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


def has_unison(df, note_i):
    n = df.loc[note_i]
    df[
        (df.pitch == n.pitch)
        & (df.release >= n.onset)
        & (df.onset <= n.release)
    ]
    return len(n) > 1
