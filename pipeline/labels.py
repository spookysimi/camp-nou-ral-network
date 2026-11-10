"""Turn tracking into one label per video frame: H (home), A (away), D (dead ball), X (abstain).

predict(tracking) is the single entry point. It works on any Tracking object, so Attila's test
bench can call exactly the same function on Metrica's sample match.

Version v0 (this file, today): the organisers' baseline_smooth50, rebuilt on our own arrays.
  1. Each frame where the ball is seen: the team of the nearest tracked player gets the ball.
     Ball not seen: nobody.
  2. Majority vote over +/- SMOOTH_WINDOW frames. Frames where nobody was seen in the whole
     window become D.

It is deliberately NOT good. Its job is to prove the pipeline end to end: if our file comes out
identical to the official baseline_smooth50.csv, the loading and frame alignment are right.
v1 replaces step 2 with a state machine built from Hugo and Nathan's possession rules.
"""

import numpy as np

from . import config
from .load import HOME, AWAY

NOBODY = -1
CHUNK = 20000  # frames per batch in nearest_team, to keep memory use modest


def nearest_team(tracking):
    """Per frame: HOME, AWAY, or NOBODY (ball not seen, or no player seen).

    Distances are measured in metres (normalised x * pitch length, y * pitch width), because the
    pitch is not square. The arithmetic is written exactly like starter/baseline.py so the result
    matches it bit for bit; on equal distances the lowest slot number wins, as there.
    """
    length, width = tracking.pitch
    n = tracking.n_frames
    out = np.full(n, NOBODY, dtype=np.int8)

    for start in range(0, n, CHUNK):
        stop = min(n, start + CHUNK)
        ball_x = tracking.ball[start:stop, 0] * length
        ball_y = tracking.ball[start:stop, 1] * width
        dx = tracking.players_xy[start:stop, :, 0] * length - ball_x[:, None]
        dy = tracking.players_xy[start:stop, :, 1] * width - ball_y[:, None]
        distance = dx * dx + dy * dy                      # NaN where player or ball unseen

        has_ball = ~np.isnan(tracking.ball[start:stop, 0])
        has_player = ~np.all(np.isnan(distance), axis=1)
        usable = has_ball & has_player

        nearest = np.argmin(np.where(np.isnan(distance), np.inf, distance), axis=1)
        out[start:stop][usable] = tracking.team_of_slot[nearest[usable]]
    return out


def smooth_majority(codes, window):
    """Centred majority vote over +/- window frames; D where nobody had the ball all window.

    Same method as starter/baseline.py smooth(): cumulative sums, so cost does not depend on
    the window size. Ties go to home.
    """
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))

    home_cum = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_cum = np.concatenate(([0], np.cumsum(codes == AWAY)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_cum[high] - home_cum[low]
    away = away_cum[high] - away_cum[low]

    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def predict(tracking, window=None):
    """One label per video frame, as a numpy array of 'H' / 'A' / 'D' / 'X'. Version v0."""
    window = config.SMOOTH_WINDOW if window is None else window
    labels = smooth_majority(nearest_team(tracking), window)
    assert len(labels) == tracking.n_frames, "one label per frame"
    return labels
