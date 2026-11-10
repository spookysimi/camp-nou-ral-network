"""Write the submission CSV, validate it with the organisers' own code, and summarise it.

Everything printed from here is a count or an average: safe to paste to Claude.
"""

import os
import sys

import numpy as np

from . import config

sys.path.insert(0, config.ROOT)
sys.path.insert(0, os.path.join(config.ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402  (the organisers' scorer)
from games import GAME, GAMES, half_video_blocks  # noqa: E402

VALID_LABELS = set("HADX")


def write_submission(labels, path=config.SUBMISSION_FILE):
    """frame,label CSV with one row per video frame (SUBMISSION.md)."""
    n_expected = GAMES[GAME]["n_video_frames"]
    if len(labels) != n_expected:
        raise ValueError("{} labels, the video has {} frames".format(len(labels), n_expected))
    unknown = set(np.unique(labels)) - VALID_LABELS
    if unknown:
        raise ValueError("labels outside H/A/D/X: {}".format(sorted(unknown)))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def validate(path):
    """Exactly the checks the official scorer runs. Returns None if valid, else the reason."""
    table = pm.read_label_csv(path)
    return pm.validate_submission(table, GAMES[GAME]["n_video_frames"])


def graded_runs():
    """The two halves as half-open [start, stop) video-frame ranges, from games.py."""
    return [(first, last + 1) for _, first, last in half_video_blocks(GAME)]


def summarise(labels):
    """Per half: possessions, their mean length, and the label mix on graded frames.

    Uses the scorer's own possession definition (maximal runs of H or of A) and its X handling
    (X is filled with the previous label), so these counts match what the scorer will see.
    """
    codes = pm.label_codes(labels)
    fps = GAMES[GAME]["fps"]
    rows = []
    for half, (start, stop) in enumerate(graded_runs(), start=1):
        runs = [(start, stop)]
        filled = pm.fill_abstentions(codes, runs)
        spells = pm.possession_spells(filled, runs)
        lengths = np.array([b - a for a, b, _ in spells], dtype=float) / fps
        window = codes[start:stop]
        rows.append({
            "half": half,
            "frames": stop - start,
            "possessions": len(spells),
            "mean_s": float(lengths.mean()) if len(lengths) else 0.0,
            "median_s": float(np.median(lengths)) if len(lengths) else 0.0,
            "under_1s": int(np.sum(lengths < 1.0)),
            "share": {name: float(np.mean(window == i)) for i, name in enumerate(pm.LABELS)},
        })
    return rows
