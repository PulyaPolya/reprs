# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#


import hashlib
import io
import logging
import math
import os
import random
import signal
import sys
import time
import zipfile
from functools import partial
from itertools import chain
from multiprocessing import Lock, Manager, Pool

import miditoolkit
import numpy as np

np.int = int  # type:ignore

LOGGER = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------
# type aliases
# ----------------------------------------------------------------------------------

TimeSigTup = tuple[int, int]

BarToken = int | None
PositionToken = int | None
InstrumentToken = int | None
PitchToken = int | None
DurationToken = int | None
VelocityToken = int | None
TimeSigToken = int | None
TempoToken = int | None
OctupleEncoding = tuple[
    BarToken,
    PositionToken,
    InstrumentToken,
    PitchToken,
    DurationToken,
    VelocityToken,
    TimeSigToken,
    TempoToken,
]

# ----------------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------------

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

DATA_ZIP: zipfile.ZipFile | None = None
OUTPUT_FILE = None

# ----------------------------------------------------------------------------------
# QUARTUPLE_STUFF
# ----------------------------------------------------------------------------------

QUART_BAR_I = 0
QUART_TIME_SIG_I = 1
QUART_POSITION_I = 2
QUART_TEMPO_I = 3

# ----------------------------------------------------------------------------------
# OCTUPLE STUFF
# ----------------------------------------------------------------------------------

TOKENS_PER_NOTE = 8

OCT_BAR_I = 0


lock_file = Lock()
lock_write = Lock()
lock_set = Lock()


# (0 Measure, 1 Pos, 2 Program, 3 Pitch, 4 Duration, 5 Velocity, 6 TimeSig, 7 Tempo)
# (Measure, TimeSig)
# (Pos, Tempo)
# Percussion: Program=128 Pitch=[128,255]


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


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, value, traceback):
        signal.alarm(0)


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


def b2e(x):
    x = max(x, MIN_TEMPO)
    x = min(x, MAX_TEMPO)
    x = x / MIN_TEMPO
    e = round(math.log2(x) * TEMPO_QUANT)
    return e


def e2b(x):
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


def writer(file_name, output_file, output_str_list):
    # note: parameter "file_name" is reserved for patching
    with open(output_file, "a") as f:
        for output_str in output_str_list:
            f.write(output_str + "\n")


def gen_dictionary(file_name):
    """Saves the token vocabulary to `file_name`."""
    num = 0
    with open(file_name, "w") as f:
        for j in range(BAR_MAX):
            print("<0-{}>".format(j), num, file=f)
        for j in range(BEAT_NOTE_FACTOR * MAX_NOTES_PER_BAR * POS_RESOLUTION):
            print("<1-{}>".format(j), num, file=f)
        for j in range(MAX_INST + 1 + 1):
            # max_inst + 1 for percussion
            print("<2-{}>".format(j), num, file=f)
        for j in range(2 * MAX_PITCH + 1 + 1):
            # max_pitch + 1 ~ 2 * max_pitch + 1 for percussion
            print("<3-{}>".format(j), num, file=f)
        for j in range(DURATION_MAX * POS_RESOLUTION):
            print("<4-{}>".format(j), num, file=f)
        for j in range(velocity_to_token(MAX_VELOCITY) + 1):
            print("<5-{}>".format(j), num, file=f)
        for j in range(len(TS_LIST)):
            print("<6-{}>".format(j), num, file=f)
        for j in range(b2e(MAX_TEMPO) + 1):
            print("<7-{}>".format(j), num, file=f)


def MIDI_to_encoding(midi_obj) -> list:
    def time_to_pos(t) -> int:
        return round(t * POS_RESOLUTION / midi_obj.ticks_per_beat)

    notes_start_pos = [
        time_to_pos(j.start) for i in midi_obj.instruments for j in i.notes
    ]
    if len(notes_start_pos) == 0:
        # Score is empty
        return list()

    # Get the maximum position
    # We truncate everything after this position
    # TODO: (Malcolm 2023-08-11) I think for long classical scores it may be worth
    #   slicing long scores into multiple segments
    max_pos = min(max(notes_start_pos) + 1, TRUNC_POS)

    # (Measure, TimeSig, Pos, Tempo)
    pos_to_info: list[list[int | None]] = [
        [None for _ in range(4)] for _ in range(max_pos)
    ]

    time_signature_changes = midi_obj.time_signature_changes
    time_signature_change_times = [
        time_to_pos(t.time) for t in midi_obj.time_signature_changes
    ] + [max_pos]
    tempo_changes = midi_obj.tempo_changes

    # TODO: (Malcolm 2023-08-11) numpy arrays would probably be more efficient

    for i, (time_sig_change1, time_sig_change2) in enumerate(
        zip(time_signature_change_times[:-1], time_signature_change_times[1:])
    ):
        for j in range(time_sig_change1, time_sig_change2):
            if j < len(pos_to_info):
                pos_to_info[j][QUART_TIME_SIG_I] = time_sig_to_token(
                    time_signature_reduce(
                        time_signature_changes[i].numerator,
                        time_signature_changes[i].denominator,
                    )
                )

    for i in range(len(tempo_changes)):
        for j in range(
            time_to_pos(tempo_changes[i].time),
            time_to_pos(tempo_changes[i + 1].time)
            if i < len(tempo_changes) - 1
            else max_pos,
        ):
            if j < len(pos_to_info):
                pos_to_info[j][3] = b2e(tempo_changes[i].tempo)

    # TODO: (Malcolm 2023-08-11) rather than setting None by iterating here we could
    #   just set defaults when initializing pos_to_info?

    # Set missing time signatures and tempi to defaults
    for j in range(len(pos_to_info)):
        if pos_to_info[j][QUART_TIME_SIG_I] is None:
            # MIDI default time signature
            pos_to_info[j][QUART_TIME_SIG_I] = time_sig_to_token(
                time_signature_reduce(4, 4)
            )
        if pos_to_info[j][QUART_TEMPO_I] is None:
            pos_to_info[j][QUART_TEMPO_I] = b2e(120.0)  # MIDI default tempo (BPM)

    cnt = 0
    bar = 0
    measure_length: int | None = None

    # Set bar and bar-position tokens
    for j, this_pos in enumerate(pos_to_info):
        ts = token_to_time_sig(this_pos[QUART_TIME_SIG_I])
        if cnt == 0:
            measure_length = ts[0] * BEAT_NOTE_FACTOR * POS_RESOLUTION // ts[1]
        this_pos[QUART_BAR_I] = bar
        this_pos[QUART_POSITION_I] = cnt
        cnt += 1
        assert measure_length is not None
        if cnt >= measure_length:
            assert (
                cnt == measure_length
            ), "invalid time signature change: pos = {}".format(j)
            cnt -= measure_length
            bar += 1

    encoding: list[OctupleEncoding] = []

    # start_distribution is calculated in order to optionally filter out notes with
    #   high-perplexity start locations
    start_distribution = [0] * POS_RESOLUTION

    for inst in midi_obj.instruments:
        for note in inst.notes:
            if time_to_pos(note.start) >= TRUNC_POS:
                continue
            start_distribution[time_to_pos(note.start) % POS_RESOLUTION] += 1
            info = pos_to_info[time_to_pos(note.start)]
            encoding.append(
                (  # type:ignore
                    info[0],
                    info[2],
                    MAX_INST + 1 if inst.is_drum else inst.program,
                    note.pitch + MAX_PITCH + 1 if inst.is_drum else note.pitch,
                    duration_to_token(time_to_pos(note.end) - time_to_pos(note.start)),
                    velocity_to_token(note.velocity),
                    info[1],
                    info[3],
                )
            )
    if len(encoding) == 0:
        return list()

    # Optionally filter high-perplexity onsets
    if FILTER_SYMBOLIC:
        tot = sum(start_distribution)
        start_ppl = 2 ** sum(
            (
                0 if x == 0 else -(x / tot) * math.log2((x / tot))
                for x in start_distribution
            )
        )
        # filter unaligned music
        assert (
            start_ppl <= FILTER_SYMBOLIC_PPL
        ), "filtered out by the symbolic filter: ppl = {:.2f}".format(start_ppl)

    encoding.sort()
    return encoding


def encoding_to_MIDI(encoding):
    # TODO: filter out non-valid notes and error handling
    bar_to_timesig = [
        list() for _ in range(max(map(lambda x: x[0], encoding)) + 1)  # type:ignore
    ]
    for i in encoding:
        bar_to_timesig[i[0]].append(i[6])
    bar_to_timesig = [
        max(set(i), key=i.count) if len(i) > 0 else None for i in bar_to_timesig
    ]
    for i in range(len(bar_to_timesig)):
        if bar_to_timesig[i] is None:
            bar_to_timesig[i] = (
                time_sig_to_token(time_signature_reduce(4, 4))
                if i == 0
                else bar_to_timesig[i - 1]
            )
    bar_to_pos: list[None | int] = [None] * len(bar_to_timesig)
    cur_pos = 0
    for i in range(len(bar_to_pos)):
        bar_to_pos[i] = cur_pos
        ts = token_to_time_sig(bar_to_timesig[i])
        measure_length = ts[0] * BEAT_NOTE_FACTOR * POS_RESOLUTION // ts[1]
        cur_pos += measure_length
    pos_to_tempo = [
        list()
        for _ in range(cur_pos + max(map(lambda x: x[1], encoding)))  # type:ignore
    ]
    for i in encoding:
        pos_to_tempo[bar_to_pos[i[0]] + i[1]].append(i[7])
    pos_to_tempo = [
        round(sum(i) / len(i)) if len(i) > 0 else None for i in pos_to_tempo
    ]
    for i in range(len(pos_to_tempo)):
        if pos_to_tempo[i] is None:
            pos_to_tempo[i] = b2e(120.0) if i == 0 else pos_to_tempo[i - 1]
    midi_obj = miditoolkit.midi.parser.MidiFile()

    def get_tick(bar, pos):
        return (bar_to_pos[bar] + pos) * midi_obj.ticks_per_beat // POS_RESOLUTION

    midi_obj.instruments = [
        miditoolkit.containers.Instrument(
            program=(0 if i == 128 else i), is_drum=(i == 128), name=str(i)
        )
        for i in range(128 + 1)
    ]
    for i in encoding:
        start = get_tick(i[0], i[1])
        program = i[2]
        pitch = i[3] - 128 if program == 128 else i[3]
        duration = get_tick(0, token_to_duration(i[4]))
        if duration == 0:
            duration = 1
        end = start + duration
        velocity = token_to_velocity(i[5])
        midi_obj.instruments[program].notes.append(
            miditoolkit.containers.Note(
                start=start, end=end, pitch=pitch, velocity=velocity
            )
        )
    midi_obj.instruments = [i for i in midi_obj.instruments if len(i.notes) > 0]
    cur_ts = None
    for i in range(len(bar_to_timesig)):
        new_ts = bar_to_timesig[i]
        if new_ts != cur_ts:
            numerator, denominator = token_to_time_sig(new_ts)
            midi_obj.time_signature_changes.append(
                miditoolkit.containers.TimeSignature(
                    numerator=numerator, denominator=denominator, time=get_tick(i, 0)
                )
            )
            cur_ts = new_ts
    cur_tp = None
    for i in range(len(pos_to_tempo)):
        new_tp = pos_to_tempo[i]
        if new_tp != cur_tp:
            tempo = e2b(new_tp)
            midi_obj.tempo_changes.append(
                miditoolkit.containers.TempoChange(tempo=tempo, time=get_tick(0, i))
            )
            cur_tp = new_tp
    return midi_obj


def get_hash(encoding):
    # add i[4] and i[5] for stricter match
    midi_tuple = tuple((i[2], i[3]) for i in encoding)
    midi_hash = hashlib.md5(str(midi_tuple).encode("ascii")).hexdigest()
    return midi_hash


def encode_file(file_name, midi_dict, zip_path, output_file):
    try_times = 10
    midi_file = None
    data_zip = zipfile.ZipFile(zip_path, "r")
    for _ in range(try_times):
        try:
            if MULTIPROCESS:
                lock_file.acquire()
            with data_zip.open(file_name) as f:
                # this may fail due to unknown bug
                midi_file = io.BytesIO(f.read())
        except BaseException as encoding:
            try_times -= 1
            time.sleep(1)
            if try_times == 0:
                LOGGER.error(
                    "ERROR(READ): " + file_name + " " + str(encoding), exc_info=True
                )
                return None
        finally:
            if MULTIPROCESS:
                lock_file.release()
    try:
        with timeout(seconds=600):
            midi_obj = miditoolkit.midi.parser.MidiFile(file=midi_file)
        # check abnormal values in parse result
        assert all(
            0 <= j.start < 2**31 and 0 <= j.end < 2**31
            for i in midi_obj.instruments
            for j in i.notes
        ), "bad note time"
        assert all(
            0 < j.numerator < 2**31 and 0 < j.denominator < 2**31
            for j in midi_obj.time_signature_changes
        ), "bad time signature value"
        assert 0 < midi_obj.ticks_per_beat < 2**31, "bad ticks per beat"
    except BaseException as encoding:
        LOGGER.error("ERROR(PARSE): " + file_name + " " + str(encoding), exc_info=True)
        return None
    midi_notes_count = sum(len(inst.notes) for inst in midi_obj.instruments)
    if midi_notes_count == 0:
        LOGGER.error("ERROR(BLANK): " + file_name, exc_info=True)
        return None
    try:
        encoding = MIDI_to_encoding(midi_obj)
        if len(encoding) == 0:
            LOGGER.error("ERROR(BLANK): " + file_name, exc_info=True)
            return None
        if TS_FILTER:
            allowed_ts = time_sig_to_token(time_signature_reduce(4, 4))
            if not all(i[6] == allowed_ts for i in encoding):
                LOGGER.error("ERROR(TSFILT): " + file_name, exc_info=True)
                return None
        if DEDUPLICATE:
            duplicated = False
            dup_file_name = ""
            midi_hash = "0" * 32
            try:
                midi_hash = get_hash(encoding)
            except BaseException as exc:
                pass
            if MULTIPROCESS:
                lock_set.acquire()
            if midi_hash in midi_dict:
                dup_file_name = midi_dict[midi_hash]
                duplicated = True
            else:
                midi_dict[midi_hash] = file_name
            if MULTIPROCESS:
                lock_set.release()
            if duplicated:
                LOGGER.error(
                    "ERROR(DUPLICATED): "
                    + midi_hash
                    + " "
                    + file_name
                    + " == "
                    + dup_file_name,
                    exc_info=True,
                )
                return None

        output_str_list: list[str] = []

        # Iterate through overlapping samples
        sample_step: int = max(round(SAMPLE_LEN_MAX / SAMPLE_OVERLAP_RATE), 1)
        for p in range(
            0 - random.randint(0, SAMPLE_LEN_MAX - 1), len(encoding), sample_step
        ):
            L = max(p, 0)
            R = min(p + SAMPLE_LEN_MAX, len(encoding)) - 1

            # Malcolm: they create `bar_index_list` below where bar is not None
            #   but I don't think bar should ever be None
            assert not any(octuple[OCT_BAR_I] is None for octuple in encoding)

            bar_index_list: list[int] = [
                encoding[i][OCT_BAR_I]
                for i in range(L, R + 1)
                if encoding[i][OCT_BAR_I] is not None
            ]

            bar_index_min = 0
            bar_index_max = 0

            if len(bar_index_list) > 0:
                bar_index_min = min(bar_index_list)
                bar_index_max = max(bar_index_list)

            # Malcolm: bar index list should be monotonic so we can just take
            #   the first and last elements?
            assert bar_index_min == bar_index_list[0]
            assert bar_index_max == bar_index_list[-1]

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
            for i in encoding[L : R + 1]:
                if i[0] is None or i[0] + bar_index_offset < BAR_MAX:
                    e_segment.append(i)
                else:
                    break

            # TOKENS_PER_NOTE = 8
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
            output_str_list.append(" ".join(output_words))

        # no empty
        if not all(len(i.split()) > TOKENS_PER_NOTE * 2 - 1 for i in output_str_list):
            LOGGER.error(
                "ERROR(ENCODE): " + file_name + " " + str(encoding)  # type:ignore
            )
            return False
        try:
            if MULTIPROCESS:
                lock_write.acquire()
            writer(file_name, output_file, output_str_list)
        except BaseException as encoding:
            LOGGER.error(
                "ERROR(WRITE): " + file_name + " " + str(encoding), exc_info=True
            )
            return False
        finally:
            if MULTIPROCESS:
                lock_write.release()
        print("SUCCESS: " + file_name + "\n", end="")
        return True
    except BaseException as encoding:
        LOGGER.error(
            "ERROR(PROCESS): " + file_name + " " + str(encoding), exc_info=True
        )
        return False
    LOGGER.error("ERROR(GENERAL): " + file_name, exc_info=True)
    return False


def encode_file_with_error_handling(file_name, *args, **kwargs):
    try:
        return encode_file(file_name, *args, **kwargs)
    except BaseException as e:
        LOGGER.error("ERROR(UNCAUGHT): " + file_name, exc_info=True)
        return False


def str_to_encoding(s):
    encoding = [int(i[3:-1]) for i in s.split() if "s" not in i]
    # TOKENS_PER_NOTE = 8
    assert len(encoding) % TOKENS_PER_NOTE == 0
    encoding = [
        tuple(encoding[i + j] for j in range(TOKENS_PER_NOTE))
        for i in range(0, len(encoding), TOKENS_PER_NOTE)
    ]
    return encoding


def encoding_to_str(e):
    bar_index_offset = 0
    p = 0
    # TOKENS_PER_NOTE = 8
    return " ".join(
        (["<s>"] * TOKENS_PER_NOTE)
        + [
            "<{}-{}>".format(j, k if j > 0 else k + bar_index_offset)
            for i in e[p : p + SAMPLE_LEN_MAX]
            if i[0] + bar_index_offset < BAR_MAX
            for j, k in enumerate(i)
        ]
        + (["</s>"] * (TOKENS_PER_NOTE - 1))
    )  # 8 - 1 for append_eos functionality of binarizer in fairseq


def main():
    if MULTIPROCESS:
        manager = Manager()
        midi_dict = manager.dict()
    else:
        midi_dict = {}
    # data_path = input("Dataset zip path: ")
    # prefix = input("OctupleMIDI output path: ")
    if os.path.exists(PREFIX):
        print("Output path {} already exists!".format(PREFIX))
    else:
        os.system("mkdir -p {}".format(PREFIX))
    DATA_ZIP = zipfile.ZipFile(DATA_PATH, "r")
    file_list = [
        n
        for n in DATA_ZIP.namelist()
        if n[-4:].lower() == ".mid" or n[-5:].lower() == ".midi"
    ]
    random.seed(SEED)
    random.shuffle(file_list)
    file_list = file_list[:MAX_FILES]

    # Save the token vocabulary
    gen_dictionary("{}/dict.txt".format(PREFIX))

    ok_cnt = 0
    all_cnt = 0

    for sp in ["train", "valid", "test"]:
        total_file_cnt = len(file_list)
        file_list_split = []
        if sp == "train":  # 98%
            file_list_split = file_list[: 98 * total_file_cnt // 100]
        if sp == "valid":  # 1%
            file_list_split = file_list[
                98 * total_file_cnt // 100 : 99 * total_file_cnt // 100
            ]
        if sp == "test":  # 1%
            file_list_split = file_list[99 * total_file_cnt // 100 :]

        output_file = "{}/midi_{}.txt".format(PREFIX, sp)
        encode_f = partial(
            encode_file_with_error_handling,
            midi_dict=midi_dict,
            zip_path=DATA_PATH,
            output_file=output_file,
        )
        if MULTIPROCESS:
            with Pool(POOL_NUM) as p:
                result = list(p.imap_unordered(encode_f, file_list_split))
        else:
            result = list(map(encode_f, file_list_split))
        all_cnt += sum((1 if i is not None else 0 for i in result))
        ok_cnt += sum((1 if i is True else 0 for i in result))

    print(
        "{}/{} ({:.2f}%) MIDI files successfully processed".format(
            ok_cnt, all_cnt, ok_cnt / all_cnt * 100
        )
    )


if __name__ == "__main__":
    main()
