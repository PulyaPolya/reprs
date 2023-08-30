import itertools as it

import matplotlib.pyplot as plt


from music_df import quantize_df
from metricker import apply_weights

from reprs.image_like import ImageLikeRepr, ImageLikeSettings


from tests.helpers_for_tests import read_humdrum, get_input_kern_paths

NOTE_NAMES = list("ABCDEFG")
NOTE_NAMES += [ch + "#" for ch in NOTE_NAMES] + [ch + "b" for ch in NOTE_NAMES]


def test_image_like(n_kern_files, plot):
    paths = get_input_kern_paths(seed=42)
    tpqs = [1, 4, 8]
    min_durs = [None, 0.125]
    # spelling_vocab = ImageLikeFeatureVocab(NOTE_NAMES)
    for i, path in enumerate(paths):
        print(f"{i + 1}/{len(paths)}: {path}")
        df = read_humdrum(path)
        apply_weights(df)
        for tpq, min_dur in it.product(tpqs, min_durs):
            settings = ImageLikeSettings(
                tpq=tpq, min_dur=min_dur, onsets="weights"
            )
            encoded = ImageLikeRepr(
                df,
                settings,
                feature_names=["spelling"],
                feature_tokens={"spelling": NOTE_NAMES},
                feature_specials={"spelling": ("<NA>", "<UNK>")},
                feature_default_index={"spelling": 1},
            )
            sub_df = df[df.type == "note"]
            if min_dur is not None:
                sub_df = sub_df[sub_df.release - sub_df.onset >= min_dur]
            sub_df = quantize_df(sub_df, tpq=tpq, ticks_out=True)
            min_pitch = settings.min_pitch
            for _, note in sub_df.iterrows():
                assert encoded.onsets[note.onset, int(note.pitch) - min_pitch]
                for i in range(note.onset, note.release):
                    assert encoded.multi_hot[i, int(note.pitch) - min_pitch]
            first_segment = True
            for segment in encoded.segment(16, 2):
                assert segment["piano_roll"].shape[0] == 16
                assert segment["spelling"].shape[0] == 16
                assert (
                    (segment["piano_roll"] == 0) == (segment["spelling"] == 0)
                ).all()
                assert (
                    segment["piano_roll"][segment["onsets"] != 0] != 0
                ).all()
                if plot and first_segment:
                    fig, ax = plt.subplots(ncols=3)
                    ax[0].imshow(segment["piano_roll"].T)
                    ax[0].set_title("piano_roll")
                    ax[1].imshow(segment["onsets"].T)
                    ax[1].set_title("onsets")
                    ax[2].imshow(segment["spelling"].T)
                    ax[2].set_title("spelling")
                    plt.show()
                first_segment = False
            assert (
                (encoded.multi_hot == 0)
                == (encoded.feature_masks["spelling"] == 0)
            ).all()
