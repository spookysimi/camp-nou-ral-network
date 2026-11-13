"""Build our submission with one command.

    python run.py                  # writes output/Camp Nou-ral Network.csv and validates it
    python run.py --rebuild-cache  # re-parse the raw tracking instead of using the cache

Every line this prints is a count, an average, a time or a check result. None of it is match
data, so the whole output is safe to paste into a chat with Claude.
"""

import argparse
import sys
import time

import numpy as np

from pipeline import config, labels, load, state_machine, submission


def step(number, title):
    print("\n[{}/5] {}".format(number, title))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="parse the raw tracking file again instead of using the cache")
    args = parser.parse_args()
    started = time.time()
    problems = []

    # 1-2. Load ------------------------------------------------------------------------------
    step(1, "Load the tracking")
    tracking, report = load.load_tracking(rebuild=args.rebuild_cache)
    print("  teams: H = {}, A = {}".format(*tracking.team_names))
    print("  track slots: {}  ({} home, {} away)".format(
        report["n_slots"], int(np.sum(tracking.team_of_slot == load.HOME)),
        int(np.sum(tracking.team_of_slot == load.AWAY))))
    print("  fields per player group: {}   fields per ball group: {}".format(
        report["player_fields"], report["ball_fields"]))
    print("  channels described in the metadata ({}):".format(len(report["channels"])))
    for channel in report["channels"]:
        print("    - {}".format(channel))

    step(2, "Check the frame alignment and the data look as documented")
    n = report["n_frames_expected"]
    print("  lines {}, bad lines {}, frame_id {}..{}, duplicates {}, out of range {}, "
          "missing {}".format(report["lines"], report["bad_lines"], report["first_id"],
                              report["last_id"], report["duplicate_ids"],
                              report["ids_out_of_range"], report["frames_missing"]))
    if load.alignment_ok(report, n):
        print("  OK: one record per video frame, numbered 1..{}".format(n))
    else:
        problems.append("frame alignment does not match the organisers' description")
        print("  PROBLEM: alignment is not one record per video frame from 1. Stop and report.")

    graded = np.zeros(tracking.n_frames, dtype=bool)
    for start, stop in submission.graded_runs():
        graded[start:stop] = True
    ball_seen = ~np.isnan(tracking.ball_xy[:, 0])
    players_seen = (~np.isnan(tracking.players_xy[:, :, 0])).sum(axis=1)
    print("  ball seen on {:.1f}% of graded frames   (DATA.md: missing on ~41%, so ~59%)".format(
        100 * ball_seen[graded].mean()))
    print("  players tracked per graded frame: mean {:.1f}   (DATA.md: ~20.5 objects)".format(
        players_seen[graded].mean()))
    (_, first_stop), (second_start, _) = submission.graded_runs()
    print("  players tracked during halftime: mean {:.1f}   (load_atd.py: almost none)".format(
        players_seen[first_stop:second_start].mean()))

    # 3. Label ------------------------------------------------------------------------------
    step(3, "Label every frame (version {})".format(config.LABELER))
    t = time.time()
    predicted, stats = labels.predict(tracking, return_stats=True)
    print("  {} labels in {:.1f} s".format(len(predicted), time.time() - t))
    if stats:
        for line in state_machine.describe_v1(predicted, stats):
            print("  " + line)

    # 4. Write and validate -----------------------------------------------------------------
    step(4, "Write the CSV and validate it with the organisers' checks")
    path = submission.write_submission(predicted)
    print("  wrote {}".format(path))
    reason = submission.validate(path)
    if reason:
        problems.append("validation failed: " + reason)
        print("  FAIL: {}".format(reason))
    else:
        print("  OK: valid submission format")

    # 5. Summary ----------------------------------------------------------------------------
    step(5, "What the file says (graded frames only)")
    print("  half  possessions  mean s  median s  under 1 s     H      A      D      X")
    for row in submission.summarise(predicted):
        share = row["share"]
        print("  {:>4}  {:>11}  {:>6.1f}  {:>8.1f}  {:>9}  {:>5.1%}  {:>5.1%}  {:>5.1%}  {:>5.1%}"
              .format(row["half"], row["possessions"], row["mean_s"], row["median_s"],
                      row["under_1s"], share["H"], share["A"], share["D"], share["X"]))

    print("\nDone in {:.0f} s.".format(time.time() - started))
    if problems:
        print("PROBLEMS:")
        for problem in problems:
            print("  - " + problem)
        return 1
    if config.LABELER == "v0":
        print("Next: python checks/compare_with_baseline.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
