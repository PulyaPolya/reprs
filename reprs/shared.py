try:
    from functools import cached_property
except ImportError:
    # python <= 3.7
    from cached_property import cached_property  # type:ignore

import typing as t
from abc import abstractmethod
from dataclasses import dataclass

from time_shifter import TimeShifter

TIME_SHIFTERS = {}


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
    def validate_corpus(self, corpus_attrs: dict[str, t.Any]) -> bool:
        raise NotImplementedError


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
