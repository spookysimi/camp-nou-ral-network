"""Sanity check: how often does a submission agree with Metrica's Smart Tagging?

    python checks/smart_tagging_agreement.py                    # our file + the official baselines
    python checks/smart_tagging_agreement.py a.csv b.csv ...    # any submission files you like

WHAT THIS IS. Smart Tagging is Metrica's automatic "match diary": 638 entries such as
"Barcelona, Build Up, from 12:03 to 12:15". A team that is building up an attack has the ball,
so a good possession file should mostly agree with these entries. This script counts how often
it does. It changes nothing: it is a speedometer, not part of the labeller.

WHAT IT IS NOT. Smart Tagging is automatic and unchecked, its windows are rough (about a second)
and phases are not possessions. A perfect file would NOT score 100%. Use it to compare versions
(did v1 make things better or worse than v0?), never as the score itself, and never to fine-tune
numbers: that is what Attila's Metrica test bench is for.

Which team should have the ball during each kind of entry:
  - attacking phases (Build Up, Progression, Creating Chances, Ball in Final Third, Ball in the
    Box, Players in the Box, Long Ball, Attacking Transition): the tagged team
  - Defensive Transition (the tagged team just LOST the ball): the other team
  - Defending in Defensive / Middle / Attacking Third: the tagged team is the one defending, so
    the other team has the ball (confirmed on 13 Nov: 85% vs 15% for the opposite reading).

MIRRORS. On our match every Defending / Defensive Transition entry is the mirror image of an
attacking entry (same window, other team): Build Up <-> Defending in Attacking Third,
Progression <-> Middle Third, Creating Chances <-> Defensive Third, Attacking <-> Defensive
Transition. They add no new information, so the headline counts ATTACKING entries only and the
defending rows are printed as a check (they should repeat the attacking numbers).
  - Set Pieces and Goals are skipped: they mark stoppages, not who has the ball. (Set pieces
    become an INPUT in v2, so they must not also be the yardstick.)

Three tables per run, one column per file:
  1. % of entries where the file agrees (most of the entry's live frames show the expected team)
  2. the same, counted frame by frame instead of entry by entry
  3. % of frames inside the entries the file calls D (dead ball). Phases happen while the ball
     is in play, so a high number means the file calls stoppages that probably are not there.
Everything printed is a count or a percentage: safe to paste to Claude.
"""

import argparse
import glob
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from pipeline import config, submission  # noqa: E402
from pipeline.load import read_metadata  # noqa: E402
from games import GAME, GAMES, require  # noqa: E402
import load_smart_tagging as smart_tagging  # noqa: E402  (the organisers' loader)

pm = submission.pm  # the organisers' scorer module, already imported by submission.py
H, A, D = pm.H, pm.A, pm.D

# Params
ATTACKING = ("build up", "progression", "creating chances", "ball in final third",
             "ball in the box", "players in the box", "long ball", "attacking transition")
DEFENDING = ("defending in defensive third", "defending in middle third",
             "defending in attacking third")
OTHER_TEAM = ("defensive transition",)
SKIPPED = ("set pieces", "goals")
NO_TEAM = ("n/a", "", "none")


def norm(text):
    """Lower case, trimmed: codes are UPPER CASE and tags Title Case in the file (DATA.md)."""
    return (text or "").strip().lower()


def team_codes():
    """Club name (lower case) -> H or A code, from the tracking metadata. Nothing typed by hand."""
    home, away = read_metadata(require(["atd_meta"], GAME)[0])["team_names"]
    return {norm(home): H, norm(away): A}, (home, away)


def load_tags(names_to_code):
    """Smart Tagging entries as dicts with code, start, stop, team (H/A/None). Fails loudly on a
    team name that matches neither club, instead of silently dropping those entries."""
    events = smart_tagging.load()
    tags, unknown = [], {}
    for event in events:
        team_text = norm(event["tags"].get("Team"))
        if team_text in NO_TEAM:
            team = None
        elif team_text in names_to_code:
            team = names_to_code[team_text]
        else:
            unknown[team_text] = unknown.get(team_text, 0) + 1
            continue
        tags.append({"code": norm(event["code"]), "start": event["start_frame"],
                     "stop": event["end_frame"], "team": team})
    if unknown:
        raise ValueError("Smart Tagging team names that match neither club in the tracking "
                         "metadata: {}".format(unknown))
    return tags, smart_tagging.check_time_base(events)["agrees"]


def row_spec():
    """[(row name, codes, reading)]; reading 'same' = tagged team has the ball, 'other' = the
    other team has it. A row with no codes is a sub-heading. The order is the printed order."""
    rows = [(code.title(), (code,), "same") for code in ATTACKING]
    rows.append(("ALL ATTACKING", ATTACKING, "same"))
    rows.append(("-- mirrors of the above (check) --", (), "same"))
    rows.append(("Defensive Transition (other)", OTHER_TEAM, "other"))
    rows += [(code.title() + " (other)", (code,), "other") for code in DEFENDING]
    rows.append(("ALL DEFENDING (other)", DEFENDING, "other"))
    rows.append(("ALL DEFENDING (same) [check]", DEFENDING, "same"))
    rows.append(("HEADLINE = ALL ATTACKING", ATTACKING, "same"))
    return rows


def expected_team(tag, reading):
    return tag["team"] if reading == "same" else (A if tag["team"] == H else H)


def measure(codes, graded, tags, wanted, reading):
    """Counts for one file and one row. codes: per-frame 0..3 with X already filled."""
    n_tags = n_live_tags = n_agree_tags = n_all_dead = 0
    frames = live = agree = dead = 0
    for tag in tags:
        if tag["code"] not in wanted or tag["team"] is None:
            continue
        window = slice(max(tag["start"], 0), min(tag["stop"], len(codes)))
        inside = codes[window][graded[window]]
        if not len(inside):
            continue  # entry lies outside the graded halves
        n_tags += 1
        team = expected_team(tag, reading)
        n_live = int(np.sum((inside == H) | (inside == A)))
        n_team = int(np.sum(inside == team))
        frames += len(inside)
        live += n_live
        agree += n_team
        dead += int(np.sum(inside == D))
        if n_live == 0:
            n_all_dead += 1
            continue
        n_live_tags += 1
        n_agree_tags += int(n_team * 2 > n_live)
    return {"n_tags": n_tags, "n_live_tags": n_live_tags, "n_agree_tags": n_agree_tags,
            "n_all_dead": n_all_dead, "frames": frames, "live": live, "agree": agree,
            "dead": dead}


def load_submission(path):
    """Per-frame codes with X filled like the scorer does, after the organisers' validation."""
    reason = submission.validate(path)
    if reason:
        raise ValueError("{} is not a valid submission: {}".format(path, reason))
    codes = pm.label_codes(pm.read_label_csv(path))
    return pm.fill_abstentions(codes, submission.graded_runs())


def pct(numerator, denominator):
    return "{:>6.1f}".format(100.0 * numerator / denominator) if denominator else "     -"


def default_files():
    files = [config.SUBMISSION_FILE] if os.path.isfile(config.SUBMISSION_FILE) else []
    return files + sorted(glob.glob(os.path.join(ROOT, "evaluation", "results", "baseline_*.csv")))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="submission CSVs (default: ours + baselines)")
    args = parser.parse_args()
    files = args.files or default_files()
    if not files:
        print("no submission files found: run python run.py first, or pass CSV paths")
        return 1

    names_to_code, (home, away) = team_codes()
    tags, time_base_ok = load_tags(names_to_code)
    print("Smart Tagging: {} entries; H = {}, A = {}; time base check: {}".format(
        len(tags), home, away, "OK" if time_base_ok else "MISMATCH (stop and report)"))

    graded = np.zeros(GAMES[GAME]["n_video_frames"], dtype=bool)
    for start, stop in submission.graded_runs():
        graded[start:stop] = True

    print("\nFiles:")
    results = []
    for number, path in enumerate(files, 1):
        print("  [{}] {}".format(number, os.path.relpath(path, ROOT)))
        codes = load_submission(path)
        results.append([measure(codes, graded, tags, wanted, reading)
                        for _, wanted, reading in row_spec()])

    columns = "".join("    [{}]".format(i) for i in range(1, len(files) + 1))
    tables = (
        ("1. Entries where the file agrees, % (entries with no live frames left out)",
         lambda r: pct(r["n_agree_tags"], r["n_live_tags"]), None),
        ("2. Agreement frame by frame, % of live frames",
         lambda r: pct(r["agree"], r["live"]), None),
        ("3. Frames inside the entries that the file calls D, % (lower is better)",
         lambda r: pct(r["dead"], r["frames"]), "n_all_dead"),
    )
    for title, cell, count_key in tables:
        print("\n" + title)
        extra = "   entries all D" if count_key else ""
        print("  {:<38} {:>4}{}{}".format("entry type", "n", columns, extra))
        for row_number, (name, wanted, _) in enumerate(row_spec()):
            if not wanted:
                print("  " + name)
                continue
            rows = [result[row_number] for result in results]
            line = "  {:<38} {:>4}".format(name, rows[0]["n_tags"])
            line += "".join(" " + cell(r) for r in rows)
            if count_key:
                line += "   " + " ".join(str(r[count_key]) for r in rows)
            print(line)

    headline = max(r[-1]["n_live_tags"] for r in results) or 1
    print("\nNoise: with about {} entries, gaps under about {:.0f} points in table 1 are not real."
          .format(headline, 200.0 * math.sqrt(0.25 / headline)))
    print("Smart Tagging is automatic and makes mistakes: compare files, do not chase 100%.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
