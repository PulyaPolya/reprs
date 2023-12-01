import logging
import math
import os
import random
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

# Based on musicbert preprocessing code
import pandas as pd
from music_df import sort_df, split_musicdf
from music_df.add_feature import (
    add_default_midi_instrument,
    add_default_velocity,
    get_bar_relative_onset,
    infer_barlines,
    make_bar_explicit,
    make_instruments_explicit,
    make_tempos_explicit,
    make_time_signatures_explicit,
    split_long_bars,
)

from reprs.shared import ReprSettingsBase

LOGGER = logging.getLogger(__name__)


@dataclass
class OctupleEncodingSettings(ReprSettingsBase):
    ticks_per_beat: int = 1

    @property
    def encode_f(self):
        return oct_encode

    @property
    def inputs_vocab(self):
        # I don't believe we will use this but it needs to be implemented for
        # compatibility with chord_tones_seqs
        return "..."

    def validate_corpus(self, corpus_attrs: dict[str, Any], corpus_name: str) -> bool:
        if corpus_attrs is None:
            LOGGER.warning(f"corpus_attrs for {corpus_name} is None")
            return False
        if not corpus_attrs.get("has_time_signatures", False):
            LOGGER.info(f"Corpus lacking time signature, validation failed")
            return False
        return True


OctupleToken = namedtuple(
    "OctupleToken",
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

OCT_BAR_I = 0
OCT_POS_I = 1
OCT_INSTRUMENT_I = 2
OCT_PITCH_I = 3
OCT_DUR_I = 4
OCT_VELOCITY_I = 5
OCT_TS_I = 6
OCT_TEMPO_I = 7

TOKENS_PER_NOTE = 8

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


def token_to_time_sig(x) -> TimeSigTup:
    return TS_LIST[x]


def duration_to_token(x):
    return DUR_ENC[x] if x < len(DUR_ENC) else DUR_ENC[-1]


def token_to_duration(x):
    return DUR_DEC[x] if x < len(DUR_DEC) else DUR_DEC[-1]


def velocity_to_token(x):
    return x // VELOCITY_QUANT


def token_to_velocity(x):
    return (x * VELOCITY_QUANT) + (VELOCITY_QUANT // 2)


def tempo_to_token(x):
    x = max(x, MIN_TEMPO)
    x = min(x, MAX_TEMPO)
    x = x / MIN_TEMPO
    e = round(math.log2(x) * TEMPO_QUANT)
    return e


def token_to_tempo(x):
    return 2 ** (x / TEMPO_QUANT) * MIN_TEMPO


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


class OctupleEncoding:
    def __init__(
        self,
        tokens: list,
        features: dict[str, list[Any]],
        onsets: list[float | Fraction],
        df_indices: list[int],
        source_id: str = "unknown",
    ):
        assert len(tokens) == len(onsets)
        for feature in features.values():
            assert len(tokens) == len(feature)

        self._tokens = tokens
        self._features = features
        self._onsets = onsets
        self._df_indices = df_indices
        self._source_id = source_id

    def segment(
        self, window_len: int, hop: int | None, start_i: int | None = None
    ) -> Iterator[dict[str, Any]]:
        if hop is None:
            hop = window_len
        encoding = self._tokens
        sample_step = hop
        SAMPLE_LEN_MAX = window_len
        if start_i is None:
            if len(encoding) > window_len:
                start_i = 0 - random.randint(0, SAMPLE_LEN_MAX - 1)
            else:
                start_i = 0
        else:
            assert start_i > 0 - SAMPLE_LEN_MAX

        for p in range(start_i, len(encoding), sample_step):
            L = max(p, 0)
            R = min(p + SAMPLE_LEN_MAX, len(encoding)) - 1

            bar_index_list: list[int] = [
                encoding[i][OCT_BAR_I] for i in range(L, R + 1)
            ]

            bar_index_min = 0
            bar_index_max = 0

            if len(bar_index_list) > 0:
                bar_index_min = bar_index_list[0]
                bar_index_max = bar_index_list[-1]

            # to make bar index distribute in [0, bar_max)
            # Malcolm: i.e., to get a uniform distribution over bar numbers
            offset_lower_bound = -bar_index_min
            offset_upper_bound = BAR_MAX - 1 - bar_index_max
            bar_index_offset = (
                random.randint(offset_lower_bound, offset_upper_bound)
                if offset_lower_bound <= offset_upper_bound
                else offset_lower_bound
            )

            e_segment = []
            feature_segments = defaultdict(list)
            segment_onset = self._onsets[L]
            segment_indices = []
            for index in range(L, R + 1):
                octuple = encoding[index]
                if (
                    octuple[OCT_BAR_I] is None
                    or octuple[OCT_BAR_I] + bar_index_offset < BAR_MAX
                ):
                    segment_indices.append(self._df_indices[index])
                    e_segment.append(octuple)
                    for name, feature in self._features.items():
                        feature_segments[name].append(feature[index])
                else:
                    break

            output_words = (
                (["<s>"] * TOKENS_PER_NOTE)
                + [
                    (
                        "<{}-{}>".format(j, k if j > 0 else k + bar_index_offset)
                        if k is not None
                        else "<unk>"
                    )
                    for octuple in e_segment
                    for j, k in enumerate(octuple)
                ]
                + (["</s>"] * (TOKENS_PER_NOTE - 1))
            )  # TOKENS_PER_NOTE - 1 for append_eos functionality of binarizer in fairseq

            # (Malcolm 2023-09-01) I'm guessing based on the above comment that we
            #   should not append eos to output_features because of the append_eos
            #   functionality of the fairseq binarizer. On the other hand if
            #   I'm ever using this with something *other* than fairseq I'll need to
            #   add it back in (and likewise above).

            output_features = {
                feature_name: ["<s>"] + feature_values
                for feature_name, feature_values in feature_segments.items()
            }

            for output_feature in output_features.values():
                assert (len(output_feature) + 1) * 8 == len(output_words) + 1

            yield {
                "input": output_words,
                "segment_onset": segment_onset,
                "df_indices": segment_indices,
                "source_id": self._source_id,
            } | output_features


def oct_encode(
    music_df: pd.DataFrame,
    settings: OctupleEncodingSettings = OctupleEncodingSettings(),
    feature_names: Sequence[str] = (),
    sort: bool = True,
    **kwargs,
) -> OctupleEncoding:
    def time_to_pos(t) -> int:
        return round(t * POS_RESOLUTION / settings.ticks_per_beat)

    def pos_to_time(p) -> float:
        return p * settings.ticks_per_beat / POS_RESOLUTION

    for kwarg in kwargs:
        LOGGER.warning(f"unused kwarg to midilike_encode '{kwarg}'")

    source_id = music_df.attrs.get("source_id", "unknown")
    if not len(music_df):
        # Score is empty
        return OctupleEncoding([], {}, [], [], source_id=source_id)

    # Not sure copying is necessary since we're assigning anyway below
    music_df = music_df.copy()

    # We create a copy of the original indices because we may add or change the order
    #   of rows below
    music_df["src_indices"] = music_df.index

    if sort:
        music_df = sort_df(music_df, inplace=False)

    # truncate df

    # I'm not sure what the motivation for truncating scores is, given that we
    #   redistribute the bar numbers to the allowed range anyways.
    music_df = music_df[music_df.onset < pos_to_time(TRUNC_POS)]

    music_df["notes_start_pos"] = music_df.onset.apply(time_to_pos)

    # Time signatures
    music_df = make_time_signatures_explicit(music_df)

    # reduce time signatures:
    for i, row in music_df[music_df.type == "time_signature"].iterrows():
        numer, denom = time_signature_reduce(row.ts_numerator, row.ts_denominator)
        # There must be a better vectorized way of doing this
        music_df.loc[i, "ts_numerator"] = numer  # type:ignore
        music_df.loc[i, "ts_denominator"] = denom  # type:ignore
        music_df.loc[
            i, "other"
        ] = f'{{"numerator": {numer}, "denominator": {denom}}}'  # type:ignore

    music_df["time_sig_token"] = music_df.apply(
        lambda row: time_sig_to_token(
            time_signature_reduce(row.ts_numerator, row.ts_denominator)
        ),
        axis=1,
    )

    # Tempos
    music_df = make_tempos_explicit(music_df, default_tempo=120.0)
    music_df["tempo_token"] = music_df.tempo.apply(tempo_to_token)

    # If there are no barlines, we infer them
    if not (music_df.type == "bar").any():
        music_df = infer_barlines(music_df)

    # Sometimes scores have bars that are longer than the notated time signature
    # We split those measures "naively" (according to the notated time signature).
    # This isn't ideal but it is required to avoid OOV positions.
    music_df = split_long_bars(music_df)

    # NB we use raw bar numbers as bar tokens
    # MusicBERT uses 0-indexed bar numbers
    music_df = make_bar_explicit(music_df, initial_bar_number=0)
    music_df = get_bar_relative_onset(music_df)

    music_df["pos_token"] = music_df.bar_relative_onset.apply(time_to_pos)

    if ((music_df["pos_token"] >= 128) & (music_df["type"] == "note")).any():
        raise ValueError

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

    # features = []
    onsets = []
    features = defaultdict(list)
    tokens: list[OctupleToken] = []

    df_dict = split_musicdf(music_df)
    df_indices = []

    for inst_tuple, sub_df in df_dict.items():
        is_drum = False  # TODO: (Malcolm 2023-08-22)
        for df_i, note in sub_df[sub_df.type == "note"].iterrows():
            octuple = OctupleToken(
                bar=int(note.bar_number),
                position=int(note.pos_token),
                instrument=int(note.midi_instrument),
                pitch=int(note.pitch + MAX_PITCH + 1 if is_drum else note.pitch),
                duration=int(note.dur_token),
                velocity=int(note.velocity_token),
                time_sig=int(note.time_sig_token),
                tempo=int(note.tempo_token),
            )
            tokens.append(octuple)
            df_indices.append(music_df.loc[df_i, "src_indices"])  # type:ignore

            for name in feature_names:
                features[name].append(note[name])
            onsets.append(note.onset)
            # features.append({name: note[name] for name in feature_names})
    if len(tokens) == 0:
        return OctupleEncoding([], {}, [], [], source_id=source_id)

    sorted_indices = sorted(list(range(len(tokens))), key=tokens.__getitem__)
    tokens = [tokens[i] for i in sorted_indices]

    df_indices = [df_indices[i] for i in sorted_indices]
    features = {
        name: [feature[i] for i in sorted_indices] for name, feature in features.items()
    }
    # encoding.sort()
    encoding = OctupleEncoding(
        tokens, features, onsets, df_indices, source_id=source_id
    )

    return encoding


def oct_decode(
    encoding: Iterable[OctupleToken],
    default_time_sig=(4, 4),
    default_tempo=120.0,
    replace_zero_duration=1 / 32,
) -> pd.DataFrame:
    # (Malcolm 2023-11-06) Instruments are not yet implemented

    bar_to_ts_indices = [
        list() for _ in range(max(map(lambda x: x[0], encoding)) + 1)  # type:ignore
    ]

    for token in encoding:
        # Start each bar with the time sig token
        bar_to_ts_indices[token[OCT_BAR_I]].append(token[OCT_TS_I])

    # Every token has a time signature, so there will be many time signatures in each
    # measure. We take the most common time signature in each measure to stand for the
    # entire measure. (Hopefully there are no mid-measure time signature changes!)
    bar_to_ts = [
        max(set(bar), key=bar.count) if len(bar) > 0 else None
        for bar in bar_to_ts_indices
    ]

    for i in range(len(bar_to_ts)):
        if bar_to_ts[i] is None:
            bar_to_ts[i] = (
                time_sig_to_token(time_signature_reduce(*default_time_sig))
                if i == 0
                else bar_to_ts[i - 1]
            )

    bar_to_pos: list[None | int] = [None] * len(bar_to_ts)
    cur_pos = 0

    for i in range(len(bar_to_pos)):
        bar_to_pos[i] = cur_pos
        ts = token_to_time_sig(bar_to_ts[i])
        measure_length = ts[0] * BEAT_NOTE_FACTOR * POS_RESOLUTION // ts[1]
        cur_pos += measure_length

    pos_to_tempo = [
        list()
        for _ in range(
            cur_pos + max(map(lambda x: x[OCT_POS_I], encoding))
        )  # type:ignore
    ]

    for i in encoding:
        pos_to_tempo[bar_to_pos[i[OCT_BAR_I]] + i[OCT_POS_I]].append(i[OCT_TEMPO_I])
    pos_to_tempo = [
        round(sum(i) / len(i)) if len(i) > 0 else None for i in pos_to_tempo
    ]
    for i in range(len(pos_to_tempo)):
        if pos_to_tempo[i] is None:
            pos_to_tempo[i] = (
                tempo_to_token(default_tempo) if i == 0 else pos_to_tempo[i - 1]
            )

    note_dicts = []

    def get_quarter_note_time(bar, pos):
        return (bar_to_pos[bar] + pos) / POS_RESOLUTION

    for token in encoding:
        start = get_quarter_note_time(token[OCT_BAR_I], token[OCT_POS_I])
        program = token[OCT_INSTRUMENT_I]
        pitch = token[OCT_PITCH_I] - 128 if program == 128 else token[OCT_PITCH_I]
        duration = get_quarter_note_time(0, token_to_duration(token[OCT_DUR_I]))
        if duration == 0:
            duration = replace_zero_duration
        end = start + duration  # type:ignore
        velocity = token_to_velocity(token[OCT_VELOCITY_I])
        note_dicts.append(
            {
                "type": "note",
                "onset": start,
                "release": end,
                "pitch": pitch,
                "velocity": velocity,
            }
        )

    ts_dicts = []
    cur_ts = None
    for i in range(len(bar_to_ts)):
        new_ts = bar_to_ts[i]
        if new_ts != cur_ts:
            numerator, denominator = token_to_time_sig(new_ts)
            ts_dicts.append(
                {
                    "type": "time_signature",
                    "onset": get_quarter_note_time(i, 0),
                    "other": {"numerator": numerator, "denominator": denominator},
                }
            )
            cur_ts = new_ts

    tempo_dicts = []
    cur_tp = None
    for i in range(len(pos_to_tempo)):
        new_tp = pos_to_tempo[i]
        if new_tp != cur_tp:
            tempo = token_to_tempo(new_tp)
            tempo_dicts.append(
                {
                    "type": "tempo",
                    "other": {"tempo": tempo},
                    "onset": get_quarter_note_time(0, i),
                }
            )
            cur_tp = new_tp

    dfs = [pd.DataFrame(d) for d in (note_dicts, ts_dicts, tempo_dicts)]
    df = pd.concat(dfs, axis=0)
    df = sort_df(df)
    return df
