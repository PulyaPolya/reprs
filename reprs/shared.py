try:
    from functools import cached_property
except ImportError:
    # python <= 3.7
    from cached_property import cached_property  # type:ignore


from dataclasses import dataclass
from time_shifter import TimeShifter

TIME_SHIFTERS = {}


@dataclass
class ReprSettings:
    min_ts_exp: int = -4
    max_ts_exp: int = 4
    salami_slice: bool = False

    @cached_property
    def time_shifter(self):
        return TimeShifter(self.min_ts_exp, self.max_ts_exp)
