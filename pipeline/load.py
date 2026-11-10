"""Read the tracking (ATD) into numpy arrays, one row per video frame.

Why not kloppy? The organisers note that loading all 199,319 frames through kloppy "costs minutes
and gigabytes" (starter/load_atd.py). The raw file is simple text, so we read it ourselves, the
same way starter/baseline.py does, and keep plain numpy arrays. The first run takes a minute or
two; after that the arrays come from a cache file in data/cache/ in a few seconds.

The raw format, per line (from starter/baseline.py):

    <frame_id>:<player>;<player>;...;<player>:<ball>

Each player and ball group is comma-separated numbers ("x,y,speed" for players), with NaN where
the object was not seen. Positions are normalised 0-1 across the pitch.

Row i of every array is VIDEO frame i, which is submission row i. The file counts frame_id from
1, so video_frame = frame_id - 1 (starter/load_atd.py).

SAFETY: nothing in this module ever prints or raises an error containing values from the data.
Errors name a line number and what was wrong, never the content. That keeps every message safe
to paste into a chat with Claude.
"""

import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from . import config

sys.path.insert(0, config.ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

# Bump this when the parser changes, so old caches are rebuilt instead of silently reused.
PARSER_VERSION = 1

HOME, AWAY = 0, 1

# Room for ball fields beyond x and y (z, speed, maybe a state flag). Unused columns are dropped.
MAX_BALL_FIELDS = 8


@dataclass
class Tracking:
    """Everything the labelling step needs, for ANY match.

    Attila's test bench builds one of these from Metrica's sample data, so the exact same
    labelling code runs on our match and on the test match.
    """
    players_xy: np.ndarray     # (n_frames, n_slots, 2) float64, normalised 0-1, NaN = not seen
    players_speed: np.ndarray  # (n_frames, n_slots) float32, m/s, NaN = not seen
    ball: np.ndarray           # (n_frames, n_ball_fields) float64; columns 0, 1 = x, y
    team_of_slot: np.ndarray   # (n_slots,) int8, HOME (0) or AWAY (1)
    team_names: tuple          # (home, away) club names, for printing only
    pitch: tuple               # (length, width) in metres
    fps: int

    @property
    def n_frames(self):
        return self.players_xy.shape[0]

    @property
    def ball_xy(self):
        return self.ball[:, :2]


# -- metadata ----------------------------------------------------------------------------------

def _tag(element):
    """XML tag without a namespace prefix."""
    return element.tag.split("}")[-1]


def read_metadata(meta_path):
    """Teams, track slots and the list of data channels, from the ATD metadata XML.

    The first <Team> is the home side (FIFA/EPTS convention, and what starter/baseline.py uses).
    Every <Player> is a track SLOT, not a person: 46 slots for 22 players, and a slot does not
    follow one player through the match.
    """
    root = ET.parse(meta_path).getroot()

    teams = [el for el in root.iter() if _tag(el) == "Team"]
    if len(teams) != 2:
        raise ValueError("metadata: expected 2 <Team> entries, found {}".format(len(teams)))
    home_id = teams[0].get("id")
    names = tuple((t.findtext("Name") or "?") for t in teams)

    slots = [el.get("teamId") for el in root.iter() if _tag(el) == "Player"]
    if not slots:
        raise ValueError("metadata: no <Player> entries")
    team_of_slot = np.array([HOME if s == home_id else AWAY for s in slots], dtype=np.int8)

    # Which channels does the file describe? Names only (schema, not data). This is how we find
    # out whether the tracking carries anything beyond positions, e.g. a ball-alive flag.
    channels = sorted({"{} {}".format(_tag(el), el.get("id") or el.get("channelId") or "")
                       .strip()
                       for el in root.iter() if _tag(el).endswith("Channel")})

    return {"team_names": names, "team_of_slot": team_of_slot, "channels": channels}


# -- raw tracking ------------------------------------------------------------------------------

def _parse_numbers(text, line_no, what):
    """Comma/semicolon separated numbers -> float array. Empty fields become NaN.

    On failure the error names the line and the part of the line, never the text itself.
    """
    pieces = text.replace(";", ",").split(",")
    try:
        return np.array([p if p.strip() else "nan" for p in pieces], dtype=np.float64)
    except ValueError:
        raise ValueError("line {}: unreadable number in the {} part".format(line_no, what)) \
            from None


def parse_raw(raw_path, n_slots, n_frames):
    """Stream the raw ATD file into arrays. Returns (arrays, report).

    report holds counts only (safe to print): lines read, frame-id range, problems found.
    """
    players_xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float64)
    players_speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball = None
    seen = np.zeros(n_frames, dtype=bool)

    report = {"lines": 0, "bad_lines": 0, "first_id": None, "last_id": None,
              "duplicate_ids": 0, "ids_out_of_range": 0,
              "player_fields": set(), "ball_fields": set()}

    with open(raw_path) as handle:
        for line_no, line in enumerate(handle, start=1):
            report["lines"] += 1
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                report["bad_lines"] += 1
                continue

            try:
                frame_id = int(parts[0])
            except ValueError:
                raise ValueError("line {}: frame id is not an integer".format(line_no)) from None
            if report["first_id"] is None:
                report["first_id"] = frame_id
            report["last_id"] = frame_id

            row = frame_id - 1  # video frame = frame_id - 1
            if not 0 <= row < n_frames:
                report["ids_out_of_range"] += 1
                continue
            if seen[row]:
                report["duplicate_ids"] += 1
            seen[row] = True

            # Players: n_slots groups of "x,y,speed". A group may be shorter (e.g. a lone "NaN"
            # for an unseen slot); missing fields become NaN.
            groups = parts[1].split(";")
            if len(groups) != n_slots:
                raise ValueError("line {}: {} player groups, metadata says {}".format(
                    line_no, len(groups), n_slots))
            field_counts = [g.count(",") + 1 for g in groups]
            if len(set(field_counts)) == 1:  # fast path: every group has the same shape
                values = _parse_numbers(parts[1], line_no, "player").reshape(
                    n_slots, field_counts[0])
            else:  # slow path: pad short groups with NaN
                width = max(field_counts)
                values = np.full((n_slots, width), np.nan)
                for slot, group in enumerate(groups):
                    parsed = _parse_numbers(group, line_no, "player")
                    values[slot, :parsed.size] = parsed
            for count in set(field_counts):
                report["player_fields"].add(count)
            if values.shape[1] >= 2:
                players_xy[row] = values[:, :2]
            if values.shape[1] >= 3:
                players_speed[row] = values[:, 2]

            # Ball: one group, at least x and y (maybe more: z, speed, ...)
            ball_values = _parse_numbers(parts[2].strip(), line_no, "ball")
            report["ball_fields"].add(ball_values.size)
            if ball is None:
                ball = np.full((n_frames, MAX_BALL_FIELDS), np.nan, dtype=np.float64)
            if ball_values.size > MAX_BALL_FIELDS:
                raise ValueError("line {}: ball has {} fields, more than the {} expected".format(
                    line_no, ball_values.size, MAX_BALL_FIELDS))
            ball[row, :ball_values.size] = ball_values

    if ball is None:
        raise ValueError("no readable lines in the tracking file")

    report["frames_missing"] = int(n_frames - seen.sum())
    report["player_fields"] = sorted(report["player_fields"])
    report["ball_fields"] = sorted(report["ball_fields"])
    ball = ball[:, :max(2, max(report["ball_fields"]))]  # drop unused spare columns
    arrays = {"players_xy": players_xy, "players_speed": players_speed, "ball": ball}
    return arrays, report


def alignment_ok(report, n_frames):
    """The organisers' alignment claim: one record per video frame, ids 1..n, nothing missing."""
    return (report["first_id"] == 1 and report["last_id"] == n_frames
            and report["lines"] == n_frames and report["bad_lines"] == 0
            and report["duplicate_ids"] == 0 and report["ids_out_of_range"] == 0
            and report["frames_missing"] == 0)


# -- cache -------------------------------------------------------------------------------------

def _cache_path(source):
    return os.path.join(config.CACHE_DIR, "atd_{}_v{}.npz".format(source, PARSER_VERSION))


def _fingerprint(path):
    """Size and modification time of the raw file: if either changes, the cache is rebuilt."""
    stat = os.stat(path)
    return np.array([stat.st_size, stat.st_mtime_ns], dtype=np.int64)


def load_tracking(source="tactical", rebuild=False, log=print):
    """The match tracking as a Tracking object, plus a report of counts (safe to print).

    First call parses the raw file (a minute or two) and writes the cache. Later calls read the
    cache unless the raw file changed or rebuild=True.
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None, source)
    n_frames = source_frames(source, GAME)
    meta = read_metadata(meta_path)
    n_slots = len(meta["team_of_slot"])

    cache = _cache_path(source)
    fingerprint = _fingerprint(raw_path)
    started = time.time()
    arrays = report = None

    if not rebuild and os.path.isfile(cache):
        with np.load(cache, allow_pickle=False) as stored:
            if np.array_equal(stored["fingerprint"], fingerprint):
                arrays = {k: stored[k] for k in ("players_xy", "players_speed", "ball")}
                report = {k: int(stored["report_" + k]) for k in
                          ("lines", "bad_lines", "first_id", "last_id", "duplicate_ids",
                           "ids_out_of_range", "frames_missing")}
                report["player_fields"] = stored["report_player_fields"].tolist()
                report["ball_fields"] = stored["report_ball_fields"].tolist()
                log("  read cached arrays in {:.1f} s".format(time.time() - started))

    if arrays is None:
        log("  parsing the raw tracking file (first run only, about 1-3 minutes) ...")
        arrays, report = parse_raw(raw_path, n_slots, n_frames)
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        np.savez(cache, fingerprint=fingerprint, **arrays,
                 **{"report_" + k: np.array(v) for k, v in report.items()})
        log("  parsed in {:.1f} s and cached for next time".format(time.time() - started))

    tracking = Tracking(players_xy=arrays["players_xy"], players_speed=arrays["players_speed"],
                        ball=arrays["ball"], team_of_slot=meta["team_of_slot"],
                        team_names=meta["team_names"], pitch=GAMES[GAME]["pitch"],
                        fps=GAMES[GAME]["fps"])
    report["channels"] = meta["channels"]
    report["n_slots"] = n_slots
    report["n_frames_expected"] = n_frames
    return tracking, report
