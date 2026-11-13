"""v1 possession labeller: a "sticky" state machine.

Plain-words version (Hugo's rules 1-5 and 11):

  1. CONTROL. On each frame we ask: is exactly one team's player within CONTROL_RADIUS_M of
     the ball? If yes, that team "controls" the ball on this frame. If both teams do, it is a
     duel. If nobody does, the ball is loose. If the tracking lost the ball, we know nothing.
  2. A SWITCH NEEDS PROOF. The team in possession keeps it until the OTHER team controls the
     ball on SWITCH_MIN_FRAMES frames, with at most SWITCH_MAX_GAP "no information" frames in
     between. One touch or a deflection is not enough.
  3. BACKDATE. When a switch is confirmed, the new possession starts at the frame where the
     new team's control BEGAN, not where the proof was complete. (Being late costs points.)
  4/5. Loose balls, duels and ball-missing frames never change the team: they keep whoever had it.
  11. Before the published 2nd-half kickoff the ball is dead (D).

Dead balls in v1 are the same as v0 (a long stretch without the ball -> D) so that the v0 -> v1
comparison measures only the new team logic. After every D stretch the machine starts fresh:
the first team to prove control gets the ball, backfilled to the start of the live stretch.

Entry point: predict_v1(tracking) -> numpy array of "H"/"A"/"D" strings, one per video frame.
It only needs these fields of the Tracking object (see pipeline/load.py):
    players_xy (frames, slots, 2) normalised 0-1, NaN = not seen
    players_speed (frames, slots)  m/s, only used if USE_SPEED_CHECK
    ball (frames, >=2)             x, y normalised [, speed], NaN = not seen
    team_of_slot (slots,)          0 = home, 1 = away
    pitch (length_m, width_m)
Nothing in this file prints data values. describe_v1() returns counts and durations only, which
are safe to share.
"""

import numpy as np

from . import config

HOME, AWAY = 0, 1
UNSEEN, LOOSE, CONTESTED = -1, 2, 3          # per-frame observations besides HOME / AWAY
DEAD, UNKNOWN = 2, -1                        # per-frame label codes besides HOME / AWAY
LETTERS = np.array(["H", "A", "D"])
CHUNK = 20000                                # frames per distance batch (keeps memory low)
BAR_ATH_FRAMES = 199319


def default_params():
    """All v1 knobs, read from config.py. Pass a modified copy to predict_v1() to experiment."""
    return {
        "control_radius_m": float(config.CONTROL_RADIUS_M),
        "use_speed_check": bool(config.USE_SPEED_CHECK),
        "speed_tolerance_ms": float(config.SPEED_TOLERANCE_MS),
        "switch_min_frames": int(config.SWITCH_MIN_FRAMES),
        "switch_max_gap": int(config.SWITCH_MAX_GAP),
        "gap_d_min_frames": int(config.GAP_D_MIN_FRAMES),
        "gap_d_margin": int(config.GAP_D_MARGIN),
    }


# -- input checks -------------------------------------------------------------------------------

def _check_tracking(tracking):
    """Fail loudly, with shapes only, if the tracking does not look like what we expect."""
    xy = tracking.players_xy
    ball = tracking.ball
    teams = np.asarray(tracking.team_of_slot)
    assert xy.ndim == 3 and xy.shape[2] == 2, "players_xy must be (frames, slots, 2), got {}".format(xy.shape)
    assert ball.ndim == 2 and ball.shape[1] >= 2, "ball must be (frames, >=2), got {}".format(ball.shape)
    assert len(ball) == len(xy), "ball has {} frames, players {}".format(len(ball), len(xy))
    assert teams.shape == (xy.shape[1],), "team_of_slot {} vs {} slots".format(teams.shape, xy.shape[1])
    assert set(np.unique(teams).tolist()) == {HOME, AWAY}, "team_of_slot must hold both 0 and 1"
    length, width = tracking.pitch
    assert length > 0 and width > 0, "pitch must be positive metres"


def match_layout(tracking):
    """(halves, forced_dead) as half-open [start, stop) frame ranges.

    For our match (199,319 frames) the halves come from games.py and rule 11 adds the
    pre-kickoff stretch as forced D. For anything else (Attila's Metrica test bench), the
    tracking object may carry its own `halves` attribute; otherwise the whole array is one half.
    """
    n = len(tracking.ball)
    if getattr(tracking, "halves", None) is not None:
        return [tuple(h) for h in tracking.halves], list(getattr(tracking, "forced_dead", []) or [])
    if n == BAR_ATH_FRAMES:
        import games  # repo root is on sys.path when run via run.py
        halves = [(first, last + 1) for _, first, last in games.half_video_blocks(games.GAME)]
        second_start = halves[1][0]
        kickoff = int(config.SECOND_HALF_KICKOFF_FRAME)
        assert halves[1][0] <= kickoff < halves[1][1], "2nd-half kickoff outside the 2nd half"
        return halves, [(second_start, kickoff)]
    return [(0, n)], []


# -- rule 1: per-frame observations -------------------------------------------------------------

def frame_observations(tracking, params):
    """Per frame: HOME / AWAY (that team controls), CONTESTED, LOOSE or UNSEEN.

    Also returns the nearest team per frame (HOME/AWAY, -1 if unknown), used only as a fallback
    for a live stretch where nobody ever proves control.
    """
    length, width = tracking.pitch
    xy, ball = tracking.players_xy, tracking.ball
    teams = np.asarray(tracking.team_of_slot)
    home_slots, away_slots = teams == HOME, teams == AWAY
    radius = params["control_radius_m"]
    use_speed = params["use_speed_check"] and ball.shape[1] >= 3

    n = len(ball)
    obs = np.full(n, UNSEEN, dtype=np.int8)
    nearest = np.full(n, -1, dtype=np.int8)
    seen_idx = np.flatnonzero(np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1]))

    for lo in range(0, len(seen_idx), CHUNK):
        idx = seen_idx[lo:lo + CHUNK]
        bx = ball[idx, 0] * length
        by = ball[idx, 1] * width
        dist = np.hypot(xy[idx, :, 0] * length - bx[:, None], xy[idx, :, 1] * width - by[:, None])
        dist = np.where(np.isfinite(dist), dist, np.inf)          # unseen player = infinitely far
        near_home = dist[:, home_slots].min(axis=1)
        near_away = dist[:, away_slots].min(axis=1)

        has_player = np.isfinite(near_home) | np.isfinite(near_away)
        nearest[idx] = np.where(has_player, np.where(near_home <= near_away, HOME, AWAY), -1)

        if use_speed:
            # Rule 1, optional: the ball must move at about the player's speed. Applied to
            # the distances, so a player who is close but running a different speed does not count.
            ball_speed = ball[idx, 2][:, None]
            gap = np.abs(tracking.players_speed[idx].astype(np.float64) - ball_speed)
            ok = ~(gap > params["speed_tolerance_ms"])            # NaN speed -> do not exclude
            dist = np.where(ok, dist, np.inf)
            near_home = dist[:, home_slots].min(axis=1)
            near_away = dist[:, away_slots].min(axis=1)

        home_in, away_in = near_home <= radius, near_away <= radius
        frame_obs = np.full(len(idx), LOOSE, dtype=np.int8)
        frame_obs[home_in & ~away_in] = HOME
        frame_obs[away_in & ~home_in] = AWAY
        frame_obs[home_in & away_in] = CONTESTED
        obs[idx] = frame_obs
    return obs, nearest


# -- dead balls (v0-equivalent for now) ---------------------------------------------------------

def gap_dead(obs, halves, params):
    """Boolean per frame: inside a long ball-unseen stretch (same rule as v0's +/-2 s window)."""
    dead = np.zeros(len(obs), dtype=bool)
    margin, min_len = params["gap_d_margin"], params["gap_d_min_frames"]
    for start, stop in halves:
        unseen = np.concatenate(([0], (obs[start:stop] == UNSEEN).astype(np.int8), [0]))
        edges = np.diff(unseen)
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if e - s >= min_len and e - margin > s + margin:
                dead[start + s + margin:start + e - margin] = True
    return dead


def _runs(mask):
    """Half-open [start, stop) runs where mask is True."""
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


# -- rules 2-5: the state machine ---------------------------------------------------------------

def run_machine(obs, start, stop, params, out, stats):
    """Label one live stretch [start, stop) in place in `out`. Starts with no team in possession."""
    min_frames, max_gap = params["switch_min_frames"], params["switch_max_gap"]
    current = UNKNOWN
    challenger, challenge_start, challenge_count, last_evidence = UNKNOWN, 0, 0, 0

    for f in range(start, stop):
        o = obs[f]
        if o == HOME or o == AWAY:
            if o == current:
                challenger = UNKNOWN                      # current team touched it: challenge off
            else:
                if challenger == o and f - last_evidence <= max_gap + 1:
                    challenge_count += 1
                else:
                    challenger, challenge_start, challenge_count = o, f, 1
                last_evidence = f
                if challenge_count >= min_frames:         # rule 2: proof complete
                    if current == UNKNOWN:
                        out[start:f] = o                  # first possession: backfill stretch head
                    else:
                        out[challenge_start:f] = o        # rule 3: backdate to start of control
                        stats["switches"] += 1
                    current = o
                    challenger = UNKNOWN
        elif challenger != UNKNOWN and f - last_evidence > max_gap:
            challenger = UNKNOWN                          # challenge went stale
        out[f] = current                                  # rules 4/5: otherwise nothing changes
    return current


def predict_v1(tracking, params=None, return_stats=False):
    """Per-frame labels ("H"/"A"/"D") for every frame of the tracking."""
    _check_tracking(tracking)
    params = dict(default_params(), **(params or {}))
    halves, forced_dead = match_layout(tracking)
    n = len(tracking.ball)
    for start, stop in halves:
        assert 0 <= start < stop <= n, "half {}-{} outside 0-{}".format(start, stop, n)

    obs, nearest = frame_observations(tracking, params)
    from_gaps = gap_dead(obs, halves, params)
    dead = from_gaps.copy()
    for start, stop in forced_dead:
        dead[start:stop] = True

    codes = np.full(n, DEAD, dtype=np.int8)               # outside the halves: D (not graded)
    stats = {"switches": 0, "live_stretches": 0, "fallback_stretches": 0, "fallback_frames": 0,
             "dead_from_gaps": int(from_gaps.sum()),
             "dead_forced": int(sum(b - a for a, b in forced_dead))}

    for h_start, h_stop in halves:
        live = ~dead[h_start:h_stop]
        for s, e in _runs(live):
            start, stop = h_start + s, h_start + e
            stats["live_stretches"] += 1
            if run_machine(obs, start, stop, params, codes, stats) == UNKNOWN:
                # Nobody ever proved control here: fall back to the team most often nearest.
                stats["fallback_stretches"] += 1
                stats["fallback_frames"] += stop - start
                near = nearest[start:stop]
                home, away = int(np.sum(near == HOME)), int(np.sum(near == AWAY))
                codes[start:stop] = DEAD if home + away == 0 else (HOME if home >= away else AWAY)

    assert not np.any(codes == UNKNOWN), "internal error: unlabelled frames left"
    labels = LETTERS[codes]
    stats["observations"] = {name: int(np.sum(obs == code)) for name, code in
                             (("home_control", HOME), ("away_control", AWAY), ("contested", CONTESTED),
                              ("loose", LOOSE), ("unseen", UNSEEN))}
    return (labels, stats) if return_stats else labels


# -- safe summary --------------------------------------------------------------------------------

def describe_v1(labels, stats, halves=None, fps=25):
    """Lines of text with counts and durations only (safe to paste to Claude).

    Per-half possession lines only when `halves` is given (run.py has its own per-half table).
    """
    lines = ["v1 machine: {} switches, {} live stretches, {} fallback stretches ({} frames)".format(
        stats["switches"], stats["live_stretches"], stats["fallback_stretches"], stats["fallback_frames"]),
        "v1 dead: {} frames from long ball gaps, {} forced (pre-kickoff)".format(
            stats["dead_from_gaps"], stats["dead_forced"])]
    total = sum(stats["observations"].values())
    lines.append("v1 per-frame observations: " + ", ".join(
        "{} {:.1f}%".format(k, 100.0 * v / total) for k, v in stats["observations"].items()))
    for number, (start, stop) in enumerate(halves or [], 1):
        half = labels[start:stop]
        bounds = np.flatnonzero(half[1:] != half[:-1]) + 1
        edges = np.concatenate(([0], bounds, [len(half)]))
        spells = [(edges[i + 1] - edges[i]) / float(fps) for i in range(len(edges) - 1)
                  if half[edges[i]] in ("H", "A")]
        if not spells:
            lines.append("half {}: no possessions".format(number))
            continue
        spells = np.array(spells)
        lines.append("half {}: {} possessions, mean {:.1f} s, median {:.1f} s, {} under 1 s | "
                     "H {:.0f}% A {:.0f}% D {:.0f}%".format(
                         number, len(spells), spells.mean(), np.median(spells), int(np.sum(spells < 1)),
                         *(100.0 * np.mean(half == x) for x in ("H", "A", "D"))))
    return lines
