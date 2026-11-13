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

# -- labelling: version v1 -----------------------------------------------------------------------
# v1 is a "sticky" state machine (pipeline/state_machine.py): possession changes only when the
# other team proves control of the ball. Every number below is a first guess, to be tuned on
# Attila's Metrica test bench, NEVER on the public 5 minutes of our match (RULES.md). Change a
# number -> log it in the Decision Log.

# Which labeller predict() uses: "v0" = nearest player + majority vote (the baseline),
# "v1" = the state machine below. Keeping v0 one switch away lets us compare and roll back.
LABELER = "v1"

# Rule 1 - control. A team "controls" the ball on a frame when one of its players is within
# this many metres of the ball and no opponent is. Both teams within it = a duel (rule 5).
CONTROL_RADIUS_M = 1.0

# Rule 1, optional part: also require |ball speed - player speed| <= SPEED_TOLERANCE_MS.
# OFF for now: we believe the ball's 3rd channel is speed in m/s, but have not verified it.
USE_SPEED_CHECK = False
SPEED_TOLERANCE_MS = 3.0

# Rule 2 - a switch needs proof: the challenging team must control the ball on at least
# SWITCH_MIN_FRAMES frames (5 frames = 0.2 s) ...
SWITCH_MIN_FRAMES = 5
# ... with no more than SWITCH_MAX_GAP frames in a row without its control in between
# (ball unseen, loose or contested). If the CURRENT team controls the ball even once, the
# challenge is cancelled. With 41% of ball positions missing, strict "5 in a row" would almost
# never fire, which is why we tolerate short gaps.
SWITCH_MAX_GAP = 10

# Dead ball for v1 = exactly what v0 did, so the v0 -> v1 comparison changes ONE thing (the
# team logic). A ball-unseen stretch of at least GAP_D_MIN_FRAMES frames is labelled D, minus
# GAP_D_MARGIN frames at each end. 101 / 50 reproduces v0's "+/-2 s with no ball" rule.
# Fede's dead_ball() replaces this in v2.
GAP_D_MIN_FRAMES = 101
GAP_D_MARGIN = 50

# Rule 11 - the 2nd half is graded from frame 110,808 but kicks off at 113,742 (published in
# DATA.md, FAQ.md and starter/load_smart_tagging.py). Everything before the kickoff is D.
SECOND_HALF_KICKOFF_FRAME = 113742
