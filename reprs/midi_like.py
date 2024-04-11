from __future__ import annotations

try:
    from time_shifter import TimeShifter
except ImportError:
    pass
else:

    import logging

    from functools import cached_property

    import math  # pylint: disable=ungrouped-imports
    import os
    import pickle
    import re
    import typing as t
    import warnings
    from collections import defaultdict, deque
    from dataclasses import dataclass
    from functools import partial
    from numbers import Number

    import pandas as pd
    from music_df import get_eligible_onsets, get_eligible_releases, sort_df

    from reprs.constants import (
        END_TOKEN,
        PAD_TOKEN,
        SPECIALS,
        START_TOKEN,
        UNKNOWN_TOKEN,
    )
    from reprs.shared import MidiLikeReprSettingsBase
    from reprs.utils import get_idx_to_item_leq, get_index_to_item_leq, get_item_leq
    from reprs.vocab import Vocab
    from reprs.writers import CSVChunkWriter

    LOGGER = logging.getLogger(__name__)

    @dataclass
    class Event:
        # helper class for midilike_encode
        type_: str
        when: float
        idx: int
        pitch: int = 0
        features: t.Optional[dict] = None
        weight: t.Optional[int] = None

        def to_token(self):
            if self.type_ == "note_on":
                return "n" + chr(self.pitch)
            elif self.type_ == "note_off":
                return "N" + chr(self.pitch)
            raise ValueError(
                f"don't know how to create token from Event of type_ '{self.type_}'"
            )

    NOTE_PATTERN = re.compile(r"^note_(?P<on_or_off>on|off)<(?P<pitch>\d+(?:\.\d+)?)>$")

    NULL_FEATURE = "na"

    # Didn't end up using this.
    # def _add_to_sorted_deque(
    #     d: deque, item: t.Any, attr_name: t.Optional[str] = None, op=operator.gt
    # ):
    #     """When maintaining a collection of "sounding notes" sorted by release time
    #     we need to frequently pop from the beginning and add to the end. deque
    #     seems the most suited data structure. However, we also need to keep the
    #     collection sorted by release time. Sorting a deque is going to be very slow,
    #     but we don't really need to sort the entire deque. We're just adding
    #     one note at a time, and most often (it would be nice to check this
    #     assumption empirically) the note is going to go at the end of the queue. So
    #     this function takes care of adding to the deque in a way that maintains
    #     sort order.

    #     >>> d = deque([0, 2, 4])
    #     >>> _add_to_sorted_deque(d, 5)
    #     >>> d
    #     deque([0, 2, 4, 5])
    #     >>> _add_to_sorted_deque(d, 3)
    #     >>> d
    #     deque([0, 2, 3, 4, 5])
    #     >>> _add_to_sorted_deque(d, -1)
    #     >>> d
    #     deque([-1, 0, 2, 3, 4, 5])

    #     >>> class Item:
    #     ...     def __init__(self, x, y):
    #     ...         self.x = x
    #     ...         self.y = y
    #     ...     def __repr__(self):
    #     ...         return f"Item({self.x}, {self.y})"
    #     >>> i1 = Item(10, 3)
    #     >>> i2 = Item(9, 4)
    #     >>> i3 = Item(8, 5)
    #     >>> d = deque([i2, i3])
    #     >>> _add_to_sorted_deque(d, i1, attr_name="y")
    #     >>> d
    #     deque([Item(10, 3), Item(9, 4), Item(8, 5)])
    #     >>> d = deque()
    #     >>> _add_to_sorted_deque(d, i2, attr_name="x")
    #     >>> _add_to_sorted_deque(d, i3, attr_name="x")
    #     >>> _add_to_sorted_deque(d, i1, attr_name="x")
    #     >>> d
    #     deque([Item(8, 5), Item(9, 4), Item(10, 3)])
    #     """
    #     if not d:
    #         d.append(item)
    #         return
    #     scrap = deque()
    #     if attr_name is None:
    #         while d and op(d[-1], item):
    #             scrap.appendleft(d.pop())
    #         d.append(item)
    #         d.extend(scrap)
    #     else:
    #         while d and op(getattr(d[-1], attr_name), getattr(item, attr_name)):
    #             scrap.appendleft(d.pop())
    #         d.append(item)
    #         d.extend(scrap)

    @dataclass(eq=True)
    class SegmentIndices:
        start_orphan_indxs: t.Tuple[int]
        start_i: int
        end_i: int
        end_orphan_indxs: t.Tuple[int]

        def __post_init__(self):
            self._len = 0
            if self.start_orphan_indxs:
                self._len += len(self.start_orphan_indxs) + 1
            self._len += self.end_i - self.start_i
            if self.end_orphan_indxs:
                self._len += len(self.end_orphan_indxs) + 1

        def __len__(self):
            return self._len

        def __getitem__(self, key):
            """
            Takes an integer representing an index into the sequence of events
            returned by this segment. Returns the integer to the corresponding event
            in the original event list.

            If the event in this segment was added (which occurs with
            "time_shift<U>" where there are orphans), returns None.
            """
            if key < 0:
                raise IndexError(f"{key} is negative")
            orig_key = key
            if self.start_orphan_indxs:
                if key < len(self.start_orphan_indxs):
                    return self.start_orphan_indxs[key]
                elif key == len(self.start_orphan_indxs):
                    return None
                key -= len(self.start_orphan_indxs) + 1
            if key < (self.end_i - self.start_i):
                return self.start_i + key
            key -= self.end_i - self.start_i
            if self.end_orphan_indxs:
                if not key:
                    return None
                key -= 1
                if key < len(self.end_orphan_indxs):
                    return self.end_orphan_indxs[key]
            raise IndexError(f"{orig_key} is out of range")

    class MIDILikeRepr:
        """
        events should be sorted by "when", then by "type_" (with note_off coming
        first), then by "note_on"
        """

        def __init__(
            self,
            note_events: t.List[Event],
            time_shifter: TimeShifter,
            end_token: bool = False,
            start_token: bool = False,
            feature_names: t.Union[None, str, t.Iterable[str]] = None,
            df=None,
            for_token_classification: bool = False,
            include_barlines: bool = False,
            min_weight: Number = float("inf"),
        ):
            self.df = df
            self.ts = time_shifter
            self._include_barlines = include_barlines
            self.events = []
            self.df_indices = []
            self.features = defaultdict(list)
            self.note_on_idx_to_repr_idx = {}
            self.note_off_idx_to_repr_idx = {}
            self.repr_idx_of_last_note_off_at_time = {}
            self.note_on_i_to_feature_i = defaultdict(dict)
            self.note_off_i_to_feature_i = defaultdict(dict)
            self.sounding_notes_at_time = {0: ()}
            self.repr_note_on_indices = []
            self.repr_note_off_indices = []
            self.repr_idx_to_note_on_idx = {}
            self.repr_idx_to_note_off_idx = {}
            # I used to have the following line setting start_note_on_i to 1
            #   if start_token was True but start_note_on_i is not an index
            #   into the repr but instead into the df, so it should be 0 at the
            #   start regardless of whether we are writing out df. Or at least
            #   so I think, and tests seem to pass.
            self.for_token_classification = for_token_classification
            self.max_error = 0
            if start_token:
                self.events.append(START_TOKEN)
                self.df_indices.append(None)
                if for_token_classification:
                    for name in feature_names:
                        self.features[name].append(NULL_FEATURE)
            now = None
            sounding_notes = set()
            time_has_shifted = True

            for note_event in note_events:
                if note_event.when != now:
                    then = now
                    now = note_event.when
                    if then is not None:
                        time_shifts, error = time_shifter(now - then)
                        if error > self.max_error:
                            self.max_error = error
                        self.events.extend(time_shifts)
                        self.df_indices.extend([None for _ in time_shifts])
                        if for_token_classification:
                            for name in feature_names:
                                self.features[name].extend(
                                    [NULL_FEATURE for _ in time_shifts]
                                )
                    time_has_shifted = True
                if note_event.type_ == "bar":
                    if include_barlines:
                        self.note_on_idx_to_repr_idx[note_event.idx] = len(self)
                        for name in feature_names:
                            # TODO rename?
                            self.note_off_i_to_feature_i[name][note_event.idx] = len(
                                self.features[name]
                            )
                            if for_token_classification:
                                self.features[name].append(NULL_FEATURE)
                        self.events.append("bar")
                        self.df_indices.append(note_event.idx)
                        if now not in self.sounding_notes_at_time:
                            # if "now" has no note_off events, we need to update
                            #   sounding pitches now
                            self.sounding_notes_at_time[now] = sounding_notes.copy()
                    continue
                if note_event.type_ == "note_off":
                    self.note_off_idx_to_repr_idx[note_event.idx] = len(self)
                    self.repr_idx_of_last_note_off_at_time[now] = len(self)
                    self.repr_note_off_indices.append(len(self))
                    self.repr_idx_to_note_off_idx[len(self)] = note_event.idx
                    for name in feature_names:
                        self.note_off_i_to_feature_i[name][note_event.idx] = len(
                            self.features[name]
                        )
                        if for_token_classification:
                            self.features[name].append(NULL_FEATURE)
                    sounding_notes.remove(note_event.idx)
                    # We only need to do this after all note_off events at
                    #   this time have occurred. But that would require looking
                    #   ahead at the upcoming note_off events to see if any
                    #   have the same time. So we just update the dict every
                    #   time we see a note_off event.
                    self.sounding_notes_at_time[now] = sounding_notes.copy()
                if note_event.type_ == "note_on":
                    if now not in self.sounding_notes_at_time:
                        # if "now" has no note_off events, we need to update
                        #   sounding pitches now
                        self.sounding_notes_at_time[now] = sounding_notes.copy()
                    self.note_on_idx_to_repr_idx[note_event.idx] = len(self)
                    self.repr_note_on_indices.append(len(self))
                    self.repr_idx_to_note_on_idx[len(self)] = note_event.idx
                    for name in feature_names:
                        self.note_on_i_to_feature_i[name][note_event.idx] = len(
                            self.features[name]
                        )
                        self.features[name].append(note_event.features[name])
                    sounding_notes.add(note_event.idx)

                self.events.append(f"{note_event.type_}<{int(note_event.pitch)}>")
                self.df_indices.append(note_event.idx)
                if note_event.type_ == "note_on" and time_has_shifted:
                    # NB we put "weight" events *after* the first note at the new
                    #   time. This is maybe suboptimal in that some notes will be
                    #   followed by "weight" events while others won't, but the
                    #   neural net should be able to handle this I expect. The
                    #   alternative is to put the "weight" event before the note on
                    #   (after any note_offs). The difficulty here is that then we
                    #   have to figure out some way of "re-inserting" the weight
                    #   event when slicing segments.
                    if (
                        note_event.weight is not None
                        and note_event.weight >= min_weight
                    ):
                        self.events.append(f"weight<{note_event.weight}>")
                        if for_token_classification:
                            for name in feature_names:
                                self.features[name].append(NULL_FEATURE)
                    time_has_shifted = False
            if end_token:
                self.events.append(END_TOKEN)
                self.df_indices.append(None)
                if for_token_classification:
                    for name in feature_names:
                        self.features[name].append(NULL_FEATURE)

            for time in self.sounding_notes_at_time:
                self.sounding_notes_at_time[time] = sorted(
                    self.sounding_notes_at_time[time]
                )

        @cached_property
        def feature_indices(self):
            """returns a list of integers that indicate where the note-ons in
            self.events are (and thus where we should expect predictions in
            predicted features, as opposed to where we should expect na)"""
            return list(self.repr_idx_to_note_on_idx.keys())

        def transpose(self, interval: int) -> t.List[str]:
            transpose = partial(
                re.sub,
                NOTE_PATTERN,
                lambda m: f"note_{m.group('on_or_off')}<{int(m.group('pitch')) + interval}>",
            )
            return list(map(transpose, self.events))

        def __len__(self):
            return len(self.events)

        @cached_property
        def eligible_onsets(self):
            return get_eligible_onsets(
                self.df,
                keep_onsets_together=True,
                notes_only=not self._include_barlines,
            )

        @cached_property
        def eligible_releases(self):
            return get_eligible_releases(self.df, keep_releases_together=True)

        def _get_feature_segment(
            self,
            name,
            repr_start_i,
            repr_end_i,
            start_orphan_indxs=(),
            end_orphan_indxs=(),
        ):
            indices = {
                i for i in self.df_indices[repr_start_i:repr_end_i] if i is not None
            }
            indices = indices.union(start_orphan_indxs)
            indices = indices.union(end_orphan_indxs)
            return self.df.loc[sorted(indices), name]

        @staticmethod
        def _get_min_window_len(min_window_len, window_len):
            if min_window_len is None:
                min_window_len = window_len // 2
            else:
                assert min_window_len <= window_len
            return min_window_len

        def _get_start_orphans(
            self, start_note_on_i
        ) -> t.Tuple[t.List[int], t.List[str]]:
            start_orphan_indxs = self.sounding_notes_at_time[
                self.df.onset.loc[start_note_on_i]
            ]
            start_orphans = (
                (
                    [
                        f"note_on<{int(self.df.loc[i, 'pitch'])}>"
                        for i in start_orphan_indxs
                    ]
                    + [self.ts.unknown_time_shift]
                )
                if start_orphan_indxs
                else []
            )
            return start_orphan_indxs, start_orphans

        def _get_end_orphan_indxs(self, end_note_off_i) -> t.List[int]:
            # _get_end_orphans is separated into two functions because we
            #   don't want to generate end_orphans within the while loop below,
            #   but only after breaking out of it
            return self.sounding_notes_at_time[self.df.release.loc[end_note_off_i]]

        def _get_end_orphans(self, end_orphan_indxs) -> t.List[str]:
            return (
                (
                    [self.ts.unknown_time_shift]
                    + [
                        f"note_off<{int(self.df.loc[i, 'pitch'])}>"
                        for i in end_orphan_indxs
                    ]
                )
                if end_orphan_indxs
                else []
            )

        def _get_features(
            self, start_orphan_indxs, start_i, end_i, end_orphan_indxs
        ) -> t.Dict[str, t.List[str]]:
            if self.for_token_classification:
                start_orphan_repr_indices = [
                    self.note_on_idx_to_repr_idx[i] for i in start_orphan_indxs
                ]
                features = {}
                for name in self.features:
                    feature = [
                        self.features[name][i] for i in start_orphan_repr_indices
                    ]
                    if start_orphan_repr_indices:
                        feature.append(NULL_FEATURE)
                    feature.extend(self.features[name][start_i:end_i])
                    if end_orphan_indxs:
                        feature.extend(
                            [NULL_FEATURE for _ in range(len(end_orphan_indxs) + 1)]
                        )
                    features[name] = feature
            else:
                features = {
                    name: self._get_feature_segment(
                        name,
                        start_i,
                        end_i,
                        start_orphan_indxs,
                        end_orphan_indxs,
                    )
                    for name in self.features
                }
            return features

        def _get_segment_indxs(
            self, start_orphan_indxs, start_i, end_i, end_orphan_indxs
        ):
            return SegmentIndices(
                tuple(self.note_on_idx_to_repr_idx[i] for i in start_orphan_indxs),
                start_i,
                end_i,
                tuple(self.note_off_idx_to_repr_idx[i] for i in end_orphan_indxs),
            )

        def _advance_start_i(self, start_i, hop, eligible_onsets) -> t.Tuple[int, int]:
            next_start_i_target = start_i + hop
            next_start_i_target_note_on = get_idx_to_item_leq(
                self.repr_note_on_indices, next_start_i_target
            )
            eligible_onsets_i = get_idx_to_item_leq(
                eligible_onsets,
                next_start_i_target_note_on,
                return_first_if_larger=True,
            )
            prev_start_i = start_i
            # if hop is short it's possible that the previous steps will have
            #   backed up to the start_i we were already at, in which case
            #   we get stuck in an infinite loop. So in that case, we forcibly
            #   advance to the next eligible_onset
            while start_i <= prev_start_i:
                start_note_on_i = eligible_onsets[eligible_onsets_i]
                start_i = self.note_on_idx_to_repr_idx[start_note_on_i]
                eligible_onsets_i += 1
            return start_note_on_i, start_i, eligible_onsets_i

        def _advance_end_i(
            self, start_i, window_len, n_start_orphans, eligible_releases
        ):
            exact_end_i = start_i + window_len - n_start_orphans
            end_i_decremented = False
            while exact_end_i > start_i:
                # We subtract 1 from exact_end_i because we are looking for
                #   the last event to *include*
                repr_i = get_item_leq(self.repr_note_off_indices, exact_end_i - 1)
                possible_note_off_j = self.repr_idx_to_note_off_idx[repr_i]
                possible_off_time = self.df.release.loc[possible_note_off_j]
                end_note_off_i = get_index_to_item_leq(
                    eligible_releases, possible_off_time
                )
                end_note_off_time = self.df.release.loc[end_note_off_i]
                end_i = self.repr_idx_of_last_note_off_at_time[end_note_off_time] + 1
                if (
                    end_i == len(self.events) - 1
                    and (end_i - start_i < window_len)
                    and not end_i_decremented
                ):
                    end_i += 1
                end_orphan_indxs = self._get_end_orphan_indxs(end_note_off_i)
                if (
                    end_i
                    + (len(end_orphan_indxs) + 1 if end_orphan_indxs else 0)
                    + n_start_orphans
                    - start_i
                    < window_len
                ):
                    break
                exact_end_i -= 1
                end_i_decremented = True
            return end_i, end_orphan_indxs

        def __call__(self):
            return {"input": self.events} | self._get_features([], None, None, [])

        def segment(
            self,
            window_len: int,
            hop: int,
            min_window_len: t.Optional[int] = None,
            allow_short_initial_window: bool = True,
            allow_short_last_window: bool = True,
            return_repr_indices: bool = False,
        ) -> t.Iterator[dict[str, t.Any]]:
            """
            Keyword args:
                allow_short_initial_window: If this is False, then if
                    len(self.events) < min_window_len, there are no segments.
                    If this is True, then in that case we return a single
                    list consisting of self.events.
            Returns:

            """
            min_window_len = self._get_min_window_len(min_window_len, window_len)

            start_i = 0
            # if we never enter the while loop below, we need to define end_i
            #   so that the allow_short_last_window condition below will execute
            #   correctly.
            end_i = 0
            eligible_onsets = self.eligible_onsets
            start_note_on_i = eligible_onsets[0]
            eligible_releases = self.eligible_releases
            eligible_onsets_i = 0
            max_eligible_onset_i = len(eligible_onsets) - 1
            if allow_short_initial_window and len(self.events) < min_window_len:
                out = {
                    "input": self.events[:],
                    "segment_onset": self.df.onset.loc[0],
                } | {name: self.features[name][:] for name in self.features}

                if return_repr_indices:
                    out["repr_indices"] = SegmentIndices((), 0, len(self.events), ())
                yield out
                return
            while start_i < len(self.events) - min_window_len:
                start_orphan_indxs, start_orphans = self._get_start_orphans(
                    start_note_on_i
                )
                end_i, end_orphan_indxs = self._advance_end_i(
                    start_i, window_len, len(start_orphans), eligible_releases
                )
                end_orphans = self._get_end_orphans(end_orphan_indxs)
                segment = start_orphans + self.events[start_i:end_i] + end_orphans
                features = self._get_features(
                    start_orphan_indxs, start_i, end_i, end_orphan_indxs
                )
                out = {
                    "input": segment,
                    "segment_onset": self.df.onset.loc[start_note_on_i],
                } | features
                if return_repr_indices:
                    out["repr_indices"] = self._get_segment_indxs(
                        start_orphan_indxs, start_i, end_i, end_orphan_indxs
                    )
                yield out
                if eligible_onsets_i == max_eligible_onset_i:
                    # if there are very many parts, it is possible to get "stuck"
                    #   at the end when there are too many events remaining
                    #   to satisfy the `while start_i < len(self.events) - min_window_len`
                    #   condition for exiting this loop, and just repeat the
                    #   max_eligible_onset indefinitely. But if we already
                    #   have the segment beginning at max_eligible_onset once,
                    #   we don't need it again, so we can break here. This does
                    #   mean that in these cases the last segment will be corrupt.
                    #   So rather than risk returning it in the
                    #   "allow_short_last_window" condition below, we return directly
                    #   here.
                    return
                start_note_on_i, start_i, eligible_onsets_i = self._advance_start_i(
                    start_i, hop, eligible_onsets
                )
            if allow_short_last_window and end_i < len(self.events):
                start_orphan_indxs, start_orphans = self._get_start_orphans(
                    start_note_on_i
                )
                features = self._get_features(start_orphan_indxs, start_i, None, [])
                out = {
                    "input": start_orphans + self.events[start_i:],
                    "segment_onset": self.df.onset.loc[start_note_on_i],
                } | features
                if return_repr_indices:
                    out["repr_indices"] = self._get_segment_indxs(
                        start_orphan_indxs, start_i, len(self.events), []
                    )
                yield out

    def midilike_encode(
        df: pd.DataFrame,
        settings: MidiLikeSettings,
        feature_names: t.Union[None, str, t.Iterable[str]] = None,
        sort: bool = True,
        **kwargs,
    ):
        """
        If the DataFrame is already sorted, then use sort=False. The sorting
        should be by onset, then by pitch, then by release.
        """
        for kwarg in kwargs:
            warnings.warn(f"unused kwarg to midilike_encode '{kwarg}'")
        if sort:
            df = sort_df(df, inplace=False)
        if settings.include_barlines:
            if "bar" not in df.type.values:
                warnings.warn(
                    "`include_barlines` is True but there are no 'bar' events in `df`"
                )
        if settings.include_metric_weights and "weight" not in df.columns:
            raise ValueError(
                "`include_metric_weights` is True but there is no "
                "`weight` column in `df`"
            )
        # Unlike the previous version of this function, this function doesn't
        #   handle splitting the dataframe.
        if isinstance(feature_names, str):
            feature_names = (feature_names,)
        elif feature_names is None:
            feature_names = ()
        # I tried using sortedcontainers.SortedList to keep the contents sorted as I
        #   went, but it was actually (slightly) slower than appending to a list and
        #   then sorting twice afterwards. I assume the reason for this is because
        #   the events are so nearly in sorted order already that the overhead of
        #   finding the insertion position for every event isn't worth it.
        note_events = []
        for idx, row in df.iterrows():
            if "type" in row.index:
                # TODO should warn if include_barlines is True and "type"
                #   is not in index
                if row.type == "bar" and settings.include_barlines:
                    note_events.append(Event("bar", row.onset, idx))
                # elif row.type == "time_signature":
                #     TODO
                if row.type != "note":
                    continue
            if row.onset == row.release:
                raise ValueError(
                    "remove zero-length notes from df before calling midilike_encode()"
                )
            features = {name: row[name] for name in feature_names if name in row}
            note_events.append(
                Event(
                    "note_on",
                    row.onset,
                    idx,
                    # some features, like "enharmonic_spelling", may be null
                    # for some examples
                    pitch=row.pitch,
                    features=features,
                    weight=int(row.weight) if settings.include_metric_weights else None,
                )
            )
            note_events.append(
                Event(
                    "note_off",
                    row.release,
                    idx,
                    pitch=row.pitch,
                )
            )
        note_events.sort(key=lambda event: event.pitch)
        note_events.sort(
            # note_off must go before bar
            key=lambda event: {"note_off": 0, "bar": 1, "note_on": 2}[event.type_]
        )
        note_events.sort(key=lambda event: event.when)

        out = MIDILikeRepr(
            note_events,
            settings.time_shifter,
            settings.end_token,
            settings.start_token,
            feature_names,
            df,
            for_token_classification=settings.for_token_classification,
            include_barlines=settings.include_barlines,
            min_weight=(
                settings.min_weight_to_encode
                if settings.include_metric_weights
                else float("inf")
            ),
        )

        # LONGTERM what are or were "phantom" notes?

        return out

    @dataclass
    class Note:
        # helper class used by midilike_decode
        onset: float
        pitch: t.Optional[float] = None
        release: t.Optional[float] = None
        type_: t.Optional[str] = "note"
        track: t.Optional[int] = None
        channel: t.Optional[int] = None
        velocity: float = 64
        other: t.Optional[t.Sequence] = None

        def __call__(self):
            return pd.Series(
                {
                    "type": self.type_,
                    "track": self.track,
                    "channel": self.channel,
                    "pitch": self.pitch,
                    "onset": self.onset,
                    "release": self.release,
                    "velocity": self.velocity,
                    "other": self.other,
                }
            )

    DUPLE_TIME_SHIFT_PATTERN = re.compile(r"^time_shift<(\d+)\^(-?\d+)>$")
    TRIPLE_TIME_SHIFT_PATTERN = re.compile(r"^time_shift<\((\d+)\^(-?\d+)\)/3>$")

    def check_time_shift(event):
        m = re.match(DUPLE_TIME_SHIFT_PATTERN, event)
        if m:
            base, exp = m.groups()
            return int(base) ** int(exp)
        m = re.match(TRIPLE_TIME_SHIFT_PATTERN, event)
        if m:
            base, exp = m.groups()
            return int(base) ** int(exp) / 3
        if event == "time_shift<U>":
            return -1

    def midilike_decode(
        events: t.Union[str, t.List[str]],
        unknown_offset: float = 1.0,
    ) -> pd.DataFrame:
        """
        "Unknown" time shift events (time_shift<U>) should occur at most once in
        events as the first and last time shifts, respectively.

        There must be at least one time_shift event in 'events'.

        >>> events = [
        ...     "note_on<60>",
        ...     "time_shift<U>",
        ...     "note_on<64>",
        ...     "time_shift<2^1>",
        ...     "note_off<60>",
        ...     "time_shift<U>",
        ...     "note_off<64>",
        ... ]
        >>> midilike_decode(events)[["pitch", "onset", "release"]]
        pitch  onset  release
        0   60.0   -1.0      2.0
        1   64.0    0.0      3.0
        """

        if isinstance(events, str):
            events = events.split(" ")
        note_ons = defaultdict(deque)
        notes = []
        note_on_pattern = re.compile(r"^note_on<(\d+(\.\d+)?)>$")
        note_off_pattern = re.compile(r"^note_off<(\d+(\.\d+)?)>$")
        for event in events:
            if event.startswith("time_shift"):
                break
        else:
            raise ValueError(
                "There must be at least one `time_shift` event in `events`"
            )
        now = -unknown_offset if (check_time_shift(event) < 0) else 0

        for event in events:
            if event in (START_TOKEN, END_TOKEN, PAD_TOKEN):
                continue
            time_shift = check_time_shift(event)
            if time_shift is not None:
                if time_shift < 0:
                    time_shift = unknown_offset
                now += time_shift
                continue
            if event == "bar":
                notes.append(Note(onset=now, type_="bar"))
                continue
            if event.startswith("weight"):
                continue
            m = re.match(note_on_pattern, event)
            if m:
                pitch = float(m.group(1))
                note_ons[pitch].append(now)
                continue
            m = re.match(note_off_pattern, event)
            if m:
                pitch = float(m.group(1))
                start = note_ons[pitch].popleft()
                notes.append(Note(onset=start, pitch=pitch, release=now))
                continue
            raise ValueError(f"{event} is not a recognized event type")

        df = pd.DataFrame(n() for n in notes)
        # df.sort_values(
        #     by=["onset", "pitch", "release"],
        #     axis=0,
        #     inplace=True,
        #     ignore_index=True,
        # )
        sort_df(df, inplace=True)
        return df

    def inputs_vocab_items(
        settings: MidiLikeSettings,
    ) -> t.List[str]:
        time_shifter = TimeShifter(
            min_exp=settings.min_ts_exp, max_exp=settings.max_ts_exp
        )
        time_shifts = list(time_shifter.get_vocabulary().keys())
        note_ons = [
            f"note_on<{pitch}>"
            for pitch in range(settings.min_pitch, settings.max_pitch + 1)
        ]
        note_offs = [
            f"note_off<{pitch}>"
            for pitch in range(settings.min_pitch, settings.max_pitch + 1)
        ]
        out = time_shifts + note_ons + note_offs
        if settings.include_metric_weights:
            out.extend(
                [f"weight<{i}>" for i in range(settings.min_weight_to_encode, 2 + 1)]
            )
        if settings.include_barlines:
            out.append("bar")
        return out

    def get_inputs_vocab(dataset_root_dir):
        vocab_path = os.path.join(dataset_root_dir, "inputs_vocab.list.pickle")
        with open(vocab_path, "rb") as inf:
            vocab_items = pickle.load(inf)
        inputs_vocab = Vocab(vocab_items, specials=SPECIALS)
        inputs_vocab.set_default_index(inputs_vocab[UNKNOWN_TOKEN])
        return inputs_vocab

    def get_target_vocab(dataset_root_dir):
        vocab_path = os.path.join(dataset_root_dir, "targets_vocab.list.pickle")
        with open(vocab_path, "rb") as inf:
            vocab_items = pickle.load(inf)
        targets_vocab = Vocab(vocab_items)
        # I don't think there should be any OOV items among the targets so we
        #   don't want to set default index (if there *is* OOV it means there
        #   is a bug and we don't want to fail silently)
        # targets_vocab.set_default_index()
        return targets_vocab

    @dataclass
    class MidiLikeSettings(MidiLikeReprSettingsBase):
        include_barlines: bool = False
        include_metric_weights: bool = False
        min_weight_to_encode: int = 0
        end_token: bool = False
        start_token: bool = False
        for_token_classification: bool = True

        @property
        def file_writer(self):
            return CSVChunkWriter

        @property
        def encode_f(self):
            return midilike_encode

        @property
        def inputs_vocab(self):
            return inputs_vocab_items(self)

        def validate_corpus(
            self, corpus_attrs: dict[str, t.Any], corpus_name: str
        ) -> bool:
            if self.include_metric_weights and not corpus_attrs.get(
                "has_weights", False
            ):
                LOGGER.info(f"Corpus lacking metric weights, validation failed")
                return False
            return True
