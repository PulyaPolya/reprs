from functools import cached_property

import typing as t
from abc import abstractmethod
from dataclasses import dataclass

try:
    from time_shifter import TimeShifter

    MIDILIKE_SUPPORTED = True
except ImportError:
    MIDILIKE_SUPPORTED = False


TIME_SHIFTERS = {}


class ReprEncodeError(Exception):
    pass


@dataclass
class ReprSettingsBase:
    pass

    @property
    @abstractmethod
    def encode_f(self) -> t.Callable[..., t.Any]:
        raise NotImplementedError

    @property
    @abstractmethod
    def inputs_vocab(self):
        raise NotImplementedError

    @abstractmethod
    def validate_corpus(self, corpus_attrs: dict[str, t.Any], corpus_name: str) -> bool:
        raise NotImplementedError


if MIDILIKE_SUPPORTED:

    @dataclass
    class MidiLikeReprSettingsBase(ReprSettingsBase):
        min_ts_exp: int = -4
        max_ts_exp: int = 4
        min_pitch: int = 21  # lowest pitch of piano
        max_pitch: int = 108  # highest pitch of piano
        salami_slice: bool = False

        data_file_ext: str = "csv"

        @cached_property
        def time_shifter(self):
            return TimeShifter(self.min_ts_exp, self.max_ts_exp)

        @property
        @abstractmethod
        def file_writer(self):
            pass
