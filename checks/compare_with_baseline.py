"""One-time check: is our v0 file identical to the organisers' baseline_smooth50.csv?

    python checks/compare_with_baseline.py

v0 is built to reproduce baseline_smooth50 exactly. If every row matches, our loader, our frame
alignment and our CSV writer are all proven correct, and every later version starts from a
trusted base. Prints counts only (safe to paste to Claude).

This check only makes sense while config.SMOOTH_WINDOW is 50 and labels.py is still v0.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from pipeline import config  # noqa: E402

OFFICIAL = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")


def main():
    for path in (config.SUBMISSION_FILE, OFFICIAL):
        if not os.path.isfile(path):
            print("missing: {}  (run python run.py first?)".format(os.path.relpath(path, ROOT)))
            return 1

    ours = pm.label_codes(pm.read_label_csv(config.SUBMISSION_FILE))
    theirs = pm.label_codes(pm.read_label_csv(OFFICIAL))
    if len(ours) != len(theirs):
        print("different lengths: ours {}, official {}".format(len(ours), len(theirs)))
        return 1

    differ = ours != theirs
    n_diff = int(differ.sum())
    print("frames compared: {}".format(len(ours)))
    print("frames that differ: {}".format(n_diff))
    if n_diff == 0:
        print("IDENTICAL: our pipeline reproduces baseline_smooth50 exactly.")
        print("Loader, frame alignment and CSV writer are verified.")
        return 0

    # Where do they differ? Counts per label pair only, no frame-level output.
    print("differences by (official -> ours):")
    for a in range(len(pm.LABELS)):
        for b in range(len(pm.LABELS)):
            count = int(np.sum(differ & (theirs == a) & (ours == b)))
            if count:
                print("  {} -> {}: {}".format(pm.LABELS[a], pm.LABELS[b], count))
    print("NOT identical. Paste this output to Claude before changing anything.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
