import librosa
import numpy as np

# def beats(localscore, period, alpha):

#     backlink = -np.ones_like(localscore)
#     cumscore = localscore

#     prange = np.arange(-2*period, -np.round(period/2)+1, dtype=int)

#     txcost = -alpha * (np.abs(np.log(prange/-period))**2)

#     for i, score_i in enumerate(localscore):
#         to,


def __beat_track_dp(localscore, period, tightness):
    """Core dynamic program for beat tracking"""

    backlink = np.zeros_like(localscore, dtype=int)
    cumscore = np.zeros_like(localscore)

    # Search range for previous beat
    window = np.arange(-2 * period, -np.round(period / 2) + 1, dtype=int)

    # Make a score window, which begins biased toward
    # start_bpm and skewed
    if tightness <= 0:
        raise ParameterError("tightness must be strictly positive")

    txwt = -tightness * (np.log(-window / period) ** 2)

    # Are we on the first beat?
    first_beat = True
    for i, score_i in enumerate(localscore):

        # Are we # reaching # back # before # time # 0?
        z_pad = np.maximum(0, min(-window[0], len(window)))

        # Search over all possible predecessors
        candidates = txwt.copy()
        candidates[z_pad:] = candidates[z_pad:] + cumscore[window[z_pad:]]

        # Find # the # best # preceding # beat
        beat_location = np.argmax(candidates)

        # Add # the # local # score
        cumscore[i] = score_i + candidates[beat_location]

        # Special case  the  first onset.
        # Stop # if # the # localscore # is # small
        if first_beat and score_i < 0.01 * localscore.max():
            backlink[i] = -1
        else:
            backlink[i] = window[beat_location]
            first_beat = False

        # Update # the # time # range
        window = window + 1

    return backlink, cumscore
