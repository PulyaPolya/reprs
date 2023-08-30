from __future__ import annotations

try:
    from functools import cached_property
except ImportError:
    # python <= 3.7
    from cached_property import cached_property  # type:ignore


from dataclasses import dataclass
from collections import defaultdict
import typing as t

import pandas as pd
import numpy as np

from music_df import (
    quantize_df,
    get_eligible_onsets,
    get_eligible_releases,
)

from reprs.shared import ReprSettings
from reprs.vocab import Vocab
from reprs.utils import get_idx_to_item_leq, get_index_to_item_leq, get_item_leq


DEFAULT_MIN_WEIGHT = -3


# class ImageLikeFeatureVocab(Vocab):
#     def __init__(self, tokens):
#         super().__init__(
#             tokens=tokens,  # specials=("<NA>", "<UNK>"), default_index=1
#         )


# TODO return scipy.sparse matrices rather than dense np arrays (or at least
#   offer a flag to this extent)


class ImageLikeRepr:
    def __init__(
        self,
        df: pd.DataFrame,
        settings: ImageLikeSettings,
        feature_names: t.Iterable[str] = (),
        feature_tokens: t.Optional[t.Dict[str, t.Sequence[str]]] = None,
        feature_specials: t.Optional[t.Dict[str, t.Sequence[str]]] = None,
        feature_default_index: t.Optional[t.Dict[str, int]] = None,
    ):
        # TODO choose quantize level dynamically?
        self.settings = settings
        self.feature_names = feature_names
        if feature_names:
            if feature_specials is None:
                feature_specials = {}
            if feature_default_index is None:
                feature_default_index = {}
            assert feature_tokens is not None
            self.feature_vocabs = {
                name: Vocab(
                    tokens,
                    specials=feature_specials.get(name, None),
                    default_index=feature_default_index.get(name, None),
                )
                for name, tokens in feature_tokens.items()
            }
        else:
            self.feature_vocabs = {}
        self.feature_masks = {}
        if settings.min_dur is not None:
            df = df[df.release - df.onset >= settings.min_dur]
        df = df[df.type == "note"].reset_index(drop=True)
        df = quantize_df(df, settings.tpq, ticks_out=True)
        self.df = df
        slices, onsets, feature_slices = self._df_to_slices(df)
        self._multi_hot = self._to_multi_hot(slices, feature_slices)
        if onsets is None:
            self._onsets = None
        else:
            self._onsets = self._to_multi_hot(
                onsets, tuples=self.settings.onsets == "weights"
            )

    def _df_to_slices(self, df: pd.DataFrame):
        def _get_empty():
            return [[] for _ in range(df.release.max())]

        out = _get_empty()
        feature_slices = defaultdict(list)
        for feature_name in self.feature_names:
            feature_slices[feature_name].extend(_get_empty())
        if self.settings.onsets:
            onsets = _get_empty()
        else:
            onsets = None
        min_pitch = self.settings.min_pitch
        weights = self.settings.onsets == "weights"
        min_weight = self.settings.min_weight
        weight_offset = 2 - min_weight
        for _, note in df.iterrows():
            # Note that there can be unisons, but in that case, each note
            #   among the unisons should have the same weight and or onset
            #   value.
            if self.settings.onsets:
                onsets[note.onset].append(
                    (
                        int(note.pitch) - min_pitch,
                        weight_offset + note.weight
                        if (weights and note.weight >= min_weight)
                        else 1,
                    )
                )
            for i in range(note.onset, note.release):
                out[i].append(int(note.pitch) - min_pitch)
                for feature_name, vocab in self.feature_vocabs.items():
                    feature_slices[feature_name][i].append(
                        vocab[note[feature_name]]
                    )
        return out, onsets, feature_slices

    def _to_multi_hot(
        self,
        slices: t.List[t.List[t.Union[int, t.Tuple[int, int]]]],
        feature_slices: t.Optional[
            t.DefaultDict[str, t.List[t.List[int]]]
        ] = None,
        tuples: bool = False,
    ) -> np.ndarray:
        out = np.zeros(
            (len(slices), self.settings.max_pitch - self.settings.min_pitch),
            dtype=int,
        )
        if feature_slices:
            for name in feature_slices:
                self.feature_masks[name] = np.zeros_like(out, dtype=int)

        for i, slice_ in enumerate(slices):
            if not slice_:
                continue
            if tuples:
                slice_, values = zip(*slice_)
                out[i][np.array(slice_)] = values
            else:
                out[i][slice_] = 1
            if feature_slices:
                for name, mask in self.feature_masks.items():
                    mask[i][slice_] = feature_slices[name][i]

        return out

    @property
    def multi_hot(self):  # pylint: disable=missing-docstring
        return self._multi_hot

    @property
    def onsets(self):  # pylint: disable=missing-docstring
        return self._onsets

    @cached_property
    def eligible_onsets(self):
        return get_eligible_onsets(
            self.df, keep_onsets_together=True, notes_only=True
        )

    @cached_property
    def eligible_releases(self):
        return get_eligible_releases(self.df, keep_releases_together=True)

    def __len__(self):
        return self._multi_hot.shape[0]

    def _advance_start_i(self, start_i, hop):
        next_target_i = start_i + hop
        eligible_i = get_idx_to_item_leq(
            self.eligible_onsets, next_target_i, return_first_if_larger=True
        )
        prev_start_i = start_i
        while start_i <= prev_start_i:
            try:
                df_i = self.eligible_onsets[eligible_i]
            except IndexError:
                return len(self.multi_hot)
            start_i = self.df.loc[df_i, "onset"]
            eligible_i += 1
        return start_i

    def _advance_end_i(self, start_i, window_len):
        if self.settings.fixed_segment_len:
            return start_i + window_len
        raise NotImplementedError()
        # target_i = start_i + window_len
        # while target_i > start_i:
        #     end_i = get_index_to_item_leq(self.eligible_releases, target_i)
        #     if end_i:
        #         pass
        #     target_i -= 1
        # return end_i

    def _return(self, start_i=None, end_i=None):
        out = {
            "piano_roll": self.multi_hot[start_i:end_i],
            "tick": start_i,
            "segment_onset": start_i / self.settings.tpq,
        } | {
            name: mask[start_i:end_i]
            for name, mask in self.feature_masks.items()
        }
        if self.settings.onsets:
            out["onsets"] = self.onsets[start_i:end_i]
        return out

    def segment(
        self,
        window_len: int,
        hop: int,
    ):
        if len(self) < window_len:
            if self.settings.allow_short_initial_window:
                yield self._return()
            return
        start_i = 0
        # if we never enter the while loop below, we need to define end_i
        #   so that the allow_short_last_window condition below will execute
        #   correctly.
        end_i = 0
        while start_i < len(self) - window_len:
            end_i = self._advance_end_i(start_i, window_len)
            yield self._return(start_i, end_i)
            start_i = self._advance_start_i(start_i, hop)
        if (
            self.settings.allow_short_last_window
            and start_i < len(self)
            and end_i < len(self)
        ):
            yield self._return(start_i)


@dataclass
class ImageLikeSettings(ReprSettings):
    tpq: int = 8
    min_dur: t.Optional[float] = None
    # onsets can be "yes" or "weights"
    onsets: t.Optional[str] = None
    min_weight: int = (
        DEFAULT_MIN_WEIGHT  # should be set to same value as metricker.Meter
    )
    # if fixed_segment_len is True, then when calling ImageLikeRepr.segment,
    #   each segment will be *exactly* window_len
    fixed_segment_len: bool = True
    # if fixed_segment_len is True then both of the below are set to False
    allow_short_initial_window: bool = True
    allow_short_last_window: bool = True
    # file_writer: t.Type = TODO

    # TODO data_file_ext: str = "npz" ?

    def __post_init__(self):
        if self.fixed_segment_len:
            self.allow_short_initial_window = False
            self.allow_short_last_window = False
        if self.onsets is not None:
            assert self.onsets in ("yes", "weights")
