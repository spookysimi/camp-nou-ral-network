"""Camp Nou-ral Network: possession pipeline for More Than a Hack 2027.

Modules, in the order the data flows through them:

    config.py      every setting and file path in one place
    load.py        raw tracking file -> numpy arrays (cached after the first run)
    labels.py      arrays -> one label per video frame (H / A / D / X)
    submission.py  labels -> CSV file, validated with the organisers' own checks

run.py at the repository root ties them together into one command.
"""
