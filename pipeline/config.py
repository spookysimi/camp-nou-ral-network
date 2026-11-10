"""Every setting and file path in one place.

If you want to change how the pipeline behaves, look here first. Each setting says what it does
and where its value came from, because the technical document has to explain every choice.
"""

import os

# -- paths -------------------------------------------------------------------------------------
# ROOT is the repository folder (the one holding run.py and games.py).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The match data lives in ROOT/data (or wherever MTAH_DATA_DIR points; games.py handles that).
# Our parsed copy of the tracking is cached next to it so the slow parse runs only once.
# data/ is in .gitignore, so neither the data nor the cache can end up on GitHub.
CACHE_DIR = os.path.join(ROOT, "data", "cache")

# Where the submission is written. output/ is in .gitignore too: the repository regenerates the
# file, it never stores it.
OUTPUT_DIR = os.path.join(ROOT, "output")

# -- submission --------------------------------------------------------------------------------
# SUBMISSION.md: name the CSV after the team, exactly as registered.
TEAM_NAME = "Camp Nou-ral Network"
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, TEAM_NAME + ".csv")

# -- labelling: version v0 -----------------------------------------------------------------------
# v0 deliberately reproduces the organisers' baseline_smooth50 (nearest player to the ball, then
# a majority vote over +/- 50 frames = +/- 2 seconds). Matching it exactly proves the loader and
# the frame alignment are right before we change any logic. v1 replaces this with a state machine.
SMOOTH_WINDOW = 50
