"""Label D: find every stoppage and say which team restarts play.  (Fede, v2; fitted in by Claude)

    from pipeline.dead_ball import dead_ball
    is_dead, restarts = dead_ball(tracking, tags)        # tags may be None

    is_dead   np.ndarray[bool], one value per video frame (199,319 for BAR-ATH)
    restarts  [(frame, team)] sorted by frame; frame = first live frame after a stoppage,
              team = "H" | "A" | None (unsure)

Inputs
------
tracking  the Tracking object from pipeline/load.py. Row i of every array is video frame i.
          Used:  tracking.ball          (n, 3)  x, y (normalised 0-1), speed (m/s); NaN if missing
                 tracking.players_xy    (n, slots, 2)  x, y normalised      } only for the
                 tracking.team_of_slot  (slots,)  0 = H, 1 = A              } restart team
                 tracking.team_names    (home, away) club names, to read Smart Tagging's Team tag
          Without players every restart team is None; without team_names the BAR-ATH names are used.
tags      None, or Smart Tagging as the list of dicts from starter/load_smart_tagging.load()
          (code, start_frame, end_frame, tags{...}), or a DataFrame with code / Type / Team and
          start_frame,end_frame or timestamp,end_timestamp. load_match_tags() loads ours.

Three sources, combined as a union of dead intervals
----------------------------------------------------
1. Tracking (always on): ball out of the pitch, or ball standing still for a few seconds.
   A missing ball is UNKNOWN, never dead: it can only bridge two pieces of dead evidence.
   (This is the fix for v1's biggest error: "ball unseen 4+ s = dead" called stoppages in
   crowded attacks, where the tracking loses the ball but play goes on.)
2. Smart Tagging SET PIECES / GOALS (optional): the tag says roughly when, the tracking says
   exactly when the ball moves again. Tag team wins when present (throw-ins are always N/A).
3. Fixed facts, passed in by the caller (state_machine.match_layout knows them): for BAR-ATH
   110,808 -> 113,742 is dead, and both half kick-offs are restarts. Nothing for other matches.

Rules followed: deterministic (no randomness), and every number in PARAMS is a prior to be tuned
on Attila's test bench, never on the public 5 minutes or on anything seen in the video.
Nothing in this file prints. speed_channel_check() returns one safe percentage.
"""

import os
import sys

import numpy as np

# ----------------------------------------------------------------------------- params
FPS = 25
PARAMS = dict(
    pitch=(105.3, 68.0),        # metres, from games.py; only for player distances
    out_margin=0.01,            # normalised; ball beyond 0 - m or 1 + m counts as out
    out_min=int(0.4 * FPS),     # out for at least this long -> a real exit, not tracking noise
    still_speed=0.5,            # m/s; ball slower than this is "still"
    still_min=int(2.0 * FPS),   # still at least this long -> set piece
    kick_speed=3.0,             # m/s; ball faster than this (inside the pitch) = play restarted
    bridge_max=int(4.0 * FPS),  # NaN/still gap allowed inside one stoppage
    restart_search=int(15 * FPS),  # how far after the dead evidence to look for the kick
    tag_lead=int(3.5 * FPS),    # tags open ~3.5 s before the restart (organisers' note)
    tag_slack=int(3.0 * FPS),   # search a bit past the tag end for the kick
    tag_back_max=int(30 * FPS), # how far back from a tagged restart the stoppage may reach
    tag_min_dead=int(1.0 * FPS),# minimum D before a tagged restart
    goal_kickoff_max=int(180 * FPS),  # goal -> kick-off must happen within this
    centre_box=(0.04, 0.06),    # |x-.5|, |y-.5| for "ball on the centre spot"
    taker_radius=3.0,           # metres; nearest player within this is the taker
    taker_window=(12, 3),       # frames before / after the restart used to vote the taker
    speed_source="channel",     # "channel" = the ball's 3rd tracking value (v2, v2.1);
                                # "positions" = computed from the ball's own movement (v2.2)
    still_window=FPS // 2,      # positions: still = moved < still_speed over +/- this (1 s total)
    kick_window=5,              # positions: kick = moved > kick_speed over the last 5 frames
)

# Fallback club names if the tracking object carries none (BAR-ATH spelling, incl. "Atletic").
DEFAULT_TEAM_NAMES = {"barcelona": "H", "atletic club": "A", "athletic club": "A"}


# ----------------------------------------------------------------------------- helpers
def _runs(mask):
    """[(start, stop)] of True runs, stop exclusive."""
    m = np.concatenate(([False], np.asarray(mask, bool), [False]))
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _player_arrays(tracking):
    """(players (n, slots, 2), teams (slots,) as "H"/"A"), or (None, None)."""
    xy = getattr(tracking, "players_xy", None)
    tm = getattr(tracking, "team_of_slot", None)
    if xy is None or tm is None:
        return None, None
    tm = np.asarray(tm)
    assert tm.shape == (xy.shape[1],), "team_of_slot {} vs {} slots".format(tm.shape, xy.shape[1])
    if tm.dtype.kind in "iub":
        tm = np.where(tm == 0, "H", "A")
    return np.asarray(xy)[..., :2], tm.astype("<U1")


def _team_names(tracking):
    """Club name (lower case) -> "H"/"A", from the tracking metadata when it has names."""
    names = getattr(tracking, "team_names", None)
    if names and len(names) == 2:
        return {str(names[0]).strip().lower(): "H", str(names[1]).strip().lower(): "A"}
    return dict(DEFAULT_TEAM_NAMES)


def _norm_tags(tags, team_names):
    """-> [dict(code, type, team, start, end)] lower-case strings, frames as ints."""
    if tags is None:
        return []
    out = []
    if hasattr(tags, "iterrows"):                       # pandas DataFrame
        for _, r in tags.iterrows():
            if "start_frame" in r:
                s, e = int(r["start_frame"]), int(r["end_frame"])
            else:
                s = int(round(r["timestamp"].total_seconds() * FPS))
                e = int(round(r["end_timestamp"].total_seconds() * FPS))
            out.append(dict(code=str(r.get("code", "")), type=str(r.get("Type", "")),
                            team=str(r.get("Team", "")), start=s, end=e))
    else:                                               # starter loader dicts
        for r in tags:
            t = r.get("tags", {}) or {}
            out.append(dict(code=str(r.get("code", "")), type=str(t.get("Type", "")),
                            team=str(t.get("Team", "")), start=int(r["start_frame"]),
                            end=int(r["end_frame"])))
    for r in out:
        for k in ("code", "type", "team"):
            r[k] = (r[k] or "").strip().lower()
        r["team"] = team_names.get(r["team"])           # None for N/A / missing
    return out


def load_match_tags():
    """Our match's Smart Tagging via the organisers' starter loader, or None if it is missing."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "starter"))
    import load_smart_tagging  # noqa: E402  (the organisers' loader)
    try:
        return load_smart_tagging.load()
    except IOError:
        return None


# ----------------------------------------------------------------------------- the detector
class _Ball:
    def __init__(self, ball, p):
        ball = np.asarray(ball, float)
        assert ball.ndim == 2 and ball.shape[1] >= 3, \
            "ball must be (frames, 3) = x, y, speed; got {}".format(ball.shape)
        self.n = len(ball)
        self.x, self.y, self.v = ball[:, 0], ball[:, 1], ball[:, 2]
        self.seen = ~np.isnan(self.x) & ~np.isnan(self.y)
        if p["speed_source"] == "positions":
            # v2.2: do not trust the 3rd channel (33% agreement on our match). Two speeds from
            # the ball's own positions: a 1 s centred one for "still" (smooths tracking jitter)
            # and a 0.2 s backward one for "kick" (finds the restart a frame or two late,
            # never early). NaN wherever the ball is missing at either end.
            v_still = self._moved(p, p["still_window"], p["still_window"])
            v_kick = self._moved(p, p["kick_window"], 0)
        elif p["speed_source"] == "channel":
            v_still = v_kick = self.v
        else:
            raise ValueError("speed_source must be 'channel' or 'positions'")
        m = p["out_margin"]
        with np.errstate(invalid="ignore"):
            self.out = self.seen & ((self.x < -m) | (self.x > 1 + m) | (self.y < -m) | (self.y > 1 + m))
            self.inside = self.seen & ~self.out
            self.still = self.inside & (v_still < p["still_speed"])
            self.kick = self.inside & (v_kick > p["kick_speed"])
        # live evidence: ball visible, inside, not still
        self.live = self.inside & ~self.still

    def _moved(self, p, back, ahead):
        """Speed in m/s from the ball's position `back` frames ago to `ahead` frames ahead."""
        length, width = p["pitch"]
        span = back + ahead
        v = np.full(self.n, np.nan)
        if span <= 0 or self.n <= span:
            return v
        dx = (self.x[span:] - self.x[:-span]) * length
        dy = (self.y[span:] - self.y[:-span]) * width
        v[back:self.n - ahead] = np.hypot(dx, dy) * FPS / span
        return v

    def first_kick(self, lo, hi):
        """First frame in [lo, hi) where the ball, inside the pitch, goes faster than kick_speed
        coming from a slower frame. None if never."""
        lo, hi = max(lo, 1), min(hi, self.n)
        if hi <= lo:
            return None
        k = self.kick[lo:hi] & ~self.kick[lo - 1:hi - 1]
        idx = np.flatnonzero(k)
        return int(lo + idx[0]) if len(idx) else None


def _tracking_stoppages(b, p):
    """Dead intervals from the tracking alone: [(start, restart)]."""
    dead_ev = b.out | b.still
    # Bridge short gaps that contain no live evidence (NaN or nothing): missing never STARTS
    # a stoppage, it only joins two pieces of dead evidence.
    runs = _runs(dead_ev)
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= p["bridge_max"] and not b.live[merged[-1][1]:s].any():
            merged[-1][1] = e
        else:
            merged.append([s, e])

    out = []
    for s, e in merged:
        longest_out = max((q - r for r, q in _runs(b.out[s:e])), default=0)
        longest_still = max((q - r for r, q in _runs(b.still[s:e])), default=0)
        if longest_out < p["out_min"] and longest_still < p["still_min"]:
            continue
        r = b.first_kick(e - 1, e + p["restart_search"])
        out.append((s, r if r is not None else e))
    return out


def _tagged_restart(b, tag, p):
    """Exact restart frame for a tag: first kick inside [start, end + slack]; fallback start + lead."""
    r = b.first_kick(tag["start"], tag["end"] + p["tag_slack"])
    return r if r is not None else min(tag["start"] + p["tag_lead"], tag["end"])


def _dead_before(b, r, p):
    """Walk back from a restart while there is no live evidence; at least tag_min_dead frames."""
    lo = max(0, r - p["tag_back_max"])
    live = np.flatnonzero(b.live[lo:r])
    s = lo + int(live[-1]) + 1 if len(live) else lo
    return max(0, min(s, r - p["tag_min_dead"]))


def _taker(b, players, teams, r, p):
    """Team of the nearest player to the ball around the restart frame, by majority vote."""
    if players is None:
        return None
    before, after = p["taker_window"]
    votes = {"H": 0, "A": 0}
    L, W = p["pitch"]
    for f in range(max(0, r - before), min(b.n, r + after)):
        if not b.seen[f]:
            continue
        d = players[f] - (b.x[f], b.y[f])
        dist = np.hypot(d[:, 0] * L, d[:, 1] * W)
        if np.all(np.isnan(dist)):
            continue
        i = int(np.nanargmin(dist))
        if dist[i] <= p["taker_radius"]:
            votes[teams[i]] += 1
    if votes["H"] == votes["A"]:
        return None
    return max(votes, key=votes.get)


# ----------------------------------------------------------------------------- public API
def dead_ball(tracking, tags=None, fixed_dead=(), kickoffs=(), params=None):
    """See module docstring. Returns (is_dead, restarts).

    fixed_dead: [(start, stop)] half-open ranges known to be dead (BAR-ATH: pre-kickoff).
    kickoffs:   [frame] known kick-off frames (BAR-ATH: 11,185 and 113,742).
    The defaults are empty so another match (the Metrica test bench) never inherits ours;
    state_machine.predict_v1 passes the BAR-ATH facts from match_layout().
    """
    p = dict(PARAMS, **(params or {}))
    p["pitch"] = tuple(getattr(tracking, "pitch", p["pitch"]))
    b = _Ball(tracking.ball, p)
    players, teams = _player_arrays(tracking)
    tag_rows = _norm_tags(tags, _team_names(tracking))

    # Each candidate: [start, restart, team, priority]; priority 2 = tag/fact, 1 = tracking.
    cands = [[s, r, None, 1] for s, r in _tracking_stoppages(b, p)]

    # Smart Tagging set pieces
    set_pieces = [t for t in tag_rows if t["code"] == "set pieces"]
    for t in set_pieces:
        r = _tagged_restart(b, t, p)
        cands.append([_dead_before(b, r, p), r, t["team"], 2])

    # Goals: dead from the ball crossing the line until the kick-off; conceding team kicks off
    ko_tags = sorted(_tagged_restart(b, t, p) for t in set_pieces if "kick off" in t["type"])
    for g in (t for t in tag_rows if t["code"] == "goals"):
        lo = max(0, g["start"] - 2 * FPS)
        over = np.flatnonzero(b.out[lo:g["end"] + 1])
        start = lo + int(over[0]) if len(over) else g["start"]
        r = next((k for k in ko_tags if start < k <= start + p["goal_kickoff_max"]), None)
        if r is None:   # no tag: ball kicked from the centre spot
            cx, cy = p["centre_box"]
            with np.errstate(invalid="ignore"):
                centre = b.kick & (np.abs(b.x - .5) < cx) & (np.abs(b.y - .5) < cy)
            hits = np.flatnonzero(centre[start + 10 * FPS:start + p["goal_kickoff_max"]])
            r = start + 10 * FPS + int(hits[0]) if len(hits) else None
        if r is not None:
            team = {"H": "A", "A": "H"}.get(g["team"])
            cands.append([start, r, team, 2])

    # Fixed facts
    for s, e in fixed_dead:
        cands.append([s, e, None, 2])
    for k in kickoffs:
        ko_team = next((t["team"] for t in set_pieces
                        if "kick off" in t["type"] and t["start"] <= k <= t["end"] + p["tag_slack"]), None)
        cands.append([max(0, k - p["tag_min_dead"]), k, ko_team, 3])   # exact frame: never moved

    # Merge overlapping intervals. Restart: highest-priority source, then the latest one.
    cands.sort(key=lambda c: (c[0], c[1]))
    merged = []
    for c in cands:
        if merged and c[0] <= merged[-1][1]:
            m = merged[-1]
            m[0] = min(m[0], c[0])
            if (c[3], c[1]) > (m[3], m[1]):
                m[1], m[3] = c[1], c[3]
                m[2] = c[2] if c[2] is not None else m[2]
            elif m[2] is None:
                m[2] = c[2]
        else:
            merged.append(list(c))

    is_dead = np.zeros(b.n, bool)
    restarts = []
    for s, r, team, _ in merged:
        is_dead[max(0, s):min(b.n, r)] = True
        if r >= b.n:
            continue
        if team is None:
            team = _taker(b, players, teams, r, p)
        restarts.append((int(r), team))
    return is_dead, restarts


def speed_channel_check(tracking, tolerance=2.0, step=5):
    """% of frames where the ball's 3rd channel agrees (within `tolerance` m/s) with the speed
    computed from the ball's own movement over `step` frames (0.2 s; one frame is too noisy).
    High (say 60%+) means the channel really is speed in m/s, which the still/kick rules above
    depend on; near 0% means it is something else and v2 must not be used. One number only:
    safe to share. None if it cannot be computed."""
    ball = np.asarray(tracking.ball, float)
    if ball.ndim != 2 or ball.shape[1] < 3 or len(ball) <= step:
        return None
    length, width = tuple(getattr(tracking, "pitch", PARAMS["pitch"]))
    fps = getattr(tracking, "fps", FPS) or FPS
    dx = (ball[step:, 0] - ball[:-step, 0]) * length
    dy = (ball[step:, 1] - ball[:-step, 1]) * width
    moved = np.hypot(dx, dy) * fps / step                 # m/s from positions
    channel = 0.5 * (ball[step:, 2] + ball[:-step, 2])
    ok = np.isfinite(moved) & np.isfinite(channel)
    if not ok.any():
        return None
    return 100.0 * float(np.mean(np.abs(moved[ok] - channel[ok]) <= tolerance))
