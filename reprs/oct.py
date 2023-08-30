import math
import os
from collections import namedtuple
from typing import Sequence

import numpy as np

# Based on musicbert preprocessing code
import pandas as pd
from music_df import split_musicdf
from music_df.add_feature import (
    add_default_midi_instrument,
    add_default_velocity,
    get_bar_relative_onset,
    make_bar_explicit,
    make_instruments_explicit,
    make_tempos_explicit,
    make_time_signatures_explicit,
)

OctupleEncoding = namedtuple(
    "OctupleEncoding",
    field_names=(
        "bar",
        "position",
        "instrument",
        "pitch",
        "duration",
        "velocity",
        "time_sig",
        "tempo",
    ),
)

TimeSigTup = tuple[int, int]

DATA_PATH = os.getenv("MUSICBERT_DATAPATH", "/Users/malcolm/tmp/output.zip")
PREFIX = os.getenv("MUSICBERT_OUTPUTPATH", "/Users/malcolm/tmp/octmidi/")

# TODO: (Malcolm 2023-08-11) add multiprocess flag

MULTIPROCESS = True
MAX_FILES: int | None = None
SEED = 42

POS_RESOLUTION = 16  # per beat (quarter note)

# TODO: (Malcolm 2023-08-11) for the purposes of classical music I think we may want
#   to increase BAR_MAX. Or else transpose later segments of the track to lie within it.
BAR_MAX = 256
VELOCITY_QUANT = 4
TEMPO_QUANT = 12  # 2 ** (1 / 12)
MIN_TEMPO = 16
MAX_TEMPO = 256
DURATION_MAX = 8  # 2 ** 8 * beat
MAX_TS_DENOMINATOR = 6  # x/1 x/2 x/4 ... x/64
MAX_NOTES_PER_BAR = 2  # 1/64 ... 128/64
BEAT_NOTE_FACTOR = 4  # In MIDI format a note is always 4 beats
DEDUPLICATE = True
FILTER_SYMBOLIC = False
FILTER_SYMBOLIC_PPL = 16
TRUNC_POS = 2**16  # approx 30 minutes (1024 measures)
SAMPLE_LEN_MAX = 1000  # window length max
SAMPLE_OVERLAP_RATE = 4
TS_FILTER = False
POOL_NUM = 24
MAX_INST = 127
MAX_PITCH = 127
MAX_VELOCITY = 127

TS_DICT: dict[TimeSigTup, int] = dict()
TS_LIST: list[TimeSigTup] = list()
for i in range(0, MAX_TS_DENOMINATOR + 1):  # 1 ~ 64
    for j in range(1, ((2**i) * MAX_NOTES_PER_BAR) + 1):
        TS_DICT[(j, 2**i)] = len(TS_DICT)
        TS_LIST.append((j, 2**i))
DUR_ENC: list[int] = list()
DUR_DEC: list[int] = list()
for i in range(DURATION_MAX):
    for j in range(POS_RESOLUTION):
        DUR_DEC.append(len(DUR_ENC))
        for k in range(2**i):
            DUR_ENC.append(len(DUR_DEC) - 1)


def time_sig_to_token(x):
    assert x in TS_DICT, "unsupported time signature: " + str(x)
    return TS_DICT[x]


def duration_to_token(x):
    return DUR_ENC[x] if x < len(DUR_ENC) else DUR_ENC[-1]


def velocity_to_token(x):
    return x // VELOCITY_QUANT


def tempo_to_token(x):
    x = max(x, MIN_TEMPO)
    x = min(x, MAX_TEMPO)
    x = x / MIN_TEMPO
    e = round(math.log2(x) * TEMPO_QUANT)
    return e


def time_signature_reduce(numerator, denominator):
    # reduction (when denominator is too large)
    while (
        denominator > 2**MAX_TS_DENOMINATOR
        and denominator % 2 == 0
        and numerator % 2 == 0
    ):
        denominator //= 2
        numerator //= 2
    # decomposition (when length of a bar exceed max_notes_per_bar)
    while numerator > MAX_NOTES_PER_BAR * denominator:
        for i in range(2, numerator + 1):
            if numerator % i == 0:
                numerator //= i
                break
    return numerator, denominator


def oct_encode(
    music_df: pd.DataFrame, ticks_per_beat: int = 1, feature_names: Sequence[str] = ()
) -> tuple[list, list]:
    def time_to_pos(t) -> int:
        return round(t * POS_RESOLUTION / ticks_per_beat)

    def pos_to_time(p) -> float:
        return p * ticks_per_beat / POS_RESOLUTION

    if not len(music_df):
        # Score is empty
        return []

    # truncate df
    # TODO: (Malcolm 2023-08-11) I think for long classical scores it may be worth
    #   slicing long scores into multiple segments
    music_df = music_df[music_df.onset < pos_to_time(TRUNC_POS)]

    # Not sure copying is necessary since we're assigning anyway below
    music_df = music_df.copy()

    music_df["notes_start_pos"] = music_df.onset.apply(time_to_pos)

    # Time signatures
    music_df = make_time_signatures_explicit(music_df)
    music_df["time_sig_token"] = music_df.apply(
        lambda row: time_sig_to_token(
            time_signature_reduce(row.ts_numerator, row.ts_denominator)
        ),
        axis=1,
    )

    # Tempos
    music_df = make_tempos_explicit(music_df, default_tempo=120.0)
    music_df["tempo_token"] = music_df.tempo.apply(tempo_to_token)

    # NB we use raw bar numbers as bar tokens
    # MusicBERT uses 0-indexed bar numbers
    music_df = make_bar_explicit(music_df, initial_bar_number=0)
    music_df = get_bar_relative_onset(music_df)

    music_df["pos_token"] = music_df.bar_relative_onset.apply(time_to_pos)

    # NB we use raw midi instrument numbers as instrument tokens
    # However, they also do something like this: MAX_INST + 1 if inst.is_drum else inst.program
    music_df = make_instruments_explicit(music_df)

    # Drop non-note events

    music_df = music_df[music_df.type == "note"]

    music_df["dur_token"] = music_df.apply(
        lambda row: duration_to_token(
            time_to_pos(row.release) - time_to_pos(row.onset)
        ),
        axis=1,
    )

    music_df = add_default_velocity(music_df)
    music_df["velocity_token"] = music_df.velocity.apply(velocity_to_token).astype(int)

    features = []
    encoding: list[OctupleEncoding] = []

    df_dict = split_musicdf(music_df)

    for inst_tuple, sub_df in df_dict.items():
        is_drum = False  # TODO: (Malcolm 2023-08-22)
        for _, note in sub_df[sub_df.type == "note"].iterrows():
            octuple = OctupleEncoding(
                bar=int(note.bar_number),
                position=note.pos_token,
                instrument=note.midi_instrument,
                pitch=int(note.pitch + MAX_PITCH + 1 if is_drum else note.pitch),
                duration=note.dur_token,
                velocity=note.velocity_token,
                time_sig=note.time_sig_token,
                tempo=note.tempo_token,
            )
            encoding.append(octuple)
            features.append({name: note[name] for name in feature_names})
    if len(encoding) == 0:
        return [], []

    indices = sorted(list(range(len(encoding))), key=encoding.__getitem__)
    encoding = [encoding[i] for i in indices]
    features = [features[i] for i in indices]
    # encoding.sort()
    return encoding, features
