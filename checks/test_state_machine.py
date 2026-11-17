"""Self-test for the v1 state machine. Uses NO match data, only made-up tracking.

    python checks/test_state_machine.py

Part 1: small hand-built cases, one per rule (deflection, backdating, gaps, duels, rule 11).
Part 2: a synthetic match (random players, scripted possessions, ball dropped on ~41% of frames
in bursts) scored with spellF1, v1 against a v0-style baseline.
Part 3: v2 dead-ball cases (pipeline/dead_ball.py): ball out, ball still, ball missing in play,
Smart Tagging restart team, rule 11 on the real layout.

Part 2 is a smoke test, NOT a tuning tool: the fake match is invented, so its scores say
"the code behaves as designed", not "this setting is better". Tuning happens on Attila's
Metrica test bench. Everything printed is synthetic and safe to share.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import state_machine as sm  # noqa: E402

PITCH = (105.3, 68.0)
FPS = 25


def make_tracking(players_xy, team_of_slot, ball, halves=None):
    n = len(ball)
    return SimpleNamespace(players_xy=players_xy, team_of_slot=np.array(team_of_slot, dtype=np.int8),
                           ball=ball, players_speed=np.zeros(players_xy.shape[:2], dtype=np.float32),
                           pitch=PITCH, fps=FPS, halves=halves if halves is not None else [(0, n)])


# -- part 1: hand-built cases -------------------------------------------------------------------

def script(segments):
    """Two players: slot 0 home at (0.3, 0.5), slot 1 away at (0.7, 0.5). Ball placed per segment:
    'H' on the home player, 'A' on the away player, 'L' loose (mid-way), 'N' unseen,
    'C' contested (both players moved onto the ball)."""
    n = sum(length for _, length in segments)
    xy = np.zeros((n, 2, 2))
    xy[:, 0] = (0.3, 0.5)
    xy[:, 1] = (0.7, 0.5)
    ball = np.full((n, 3), np.nan)
    f = 0
    for kind, length in segments:
        sl = slice(f, f + length)
        if kind == "H":
            ball[sl, :2] = (0.3, 0.5)
        elif kind == "A":
            ball[sl, :2] = (0.7, 0.5)
        elif kind == "L":
            ball[sl, :2] = (0.5, 0.5)
        elif kind == "C":
            ball[sl, :2] = (0.5, 0.5)
            xy[sl, 0] = (0.5, 0.5)
            xy[sl, 1] = (0.5, 0.5)
        f += length
    return make_tracking(xy, [0, 1], ball)


def first_frame_of(labels, letter, after=0):
    hits = np.flatnonzero(labels[after:] == letter)
    return int(hits[0]) + after if len(hits) else None


def part1():
    cases = []

    labels = sm.predict_v1(script([("H", 50), ("A", 3), ("H", 50)]))
    cases.append(("deflection: 3 frames of opponent control do not switch", set(labels) == {"H"}))

    labels = sm.predict_v1(script([("H", 50), ("L", 20), ("A", 60)]))
    cases.append(("real switch is backdated to the first frame of control",
                  first_frame_of(labels, "A") == 70 and labels[69] == "H"))

    labels = sm.predict_v1(script([("H", 50), ("A", 3), ("N", 4), ("A", 3), ("H", 30)]))
    cases.append(("unseen frames inside a challenge do not break it", first_frame_of(labels, "A") == 50))

    labels = sm.predict_v1(script([("H", 50), ("A", 3), ("N", 12), ("A", 3), ("H", 30)]))
    cases.append(("a challenge expires after SWITCH_MAX_GAP frames", set(labels) == {"H"}))

    labels = sm.predict_v1(script([("H", 50), ("A", 3), ("H", 1), ("A", 3), ("H", 30)]))
    cases.append(("own-team control cancels a challenge", set(labels) == {"H"}))

    labels = sm.predict_v1(script([("H", 50), ("C", 40), ("H", 30)]))
    cases.append(("a duel (both teams close) keeps the current team", set(labels) == {"H"}))

    labels = sm.predict_v1(script([("L", 30), ("A", 40)]))
    cases.append(("first possession is backfilled to the start of the live stretch", set(labels) == {"A"}))

    labels = sm.predict_v1(script([("H", 50), ("N", 300), ("A", 50)]))
    d = np.flatnonzero(labels == "D")
    cases.append(("long ball gap -> D with 50-frame margins, fresh team after",
                  len(d) == 200 and d[0] == 100 and d[-1] == 299 and labels[300] == "A" and labels[99] == "H"))

    labels = sm.predict_v1(script([("H", 50), ("N", 100), ("H", 50)]))
    cases.append(("a 100-frame gap (below 101) is filled, not D", set(labels) == {"H"}))

    # Rule 11 on the real match layout (199,319 frames, needs games.py). Ball never seen.
    n = sm.BAR_ATH_FRAMES
    t = make_tracking(np.full((n, 2, 2), np.nan), [0, 1], np.full((n, 3), np.nan), halves=None)
    t.halves = None
    labels, stats = sm.predict_v1(t, return_stats=True)
    cases.append(("BAR-ATH layout: pre-kickoff D is 2,934 frames, output length 199,319",
                  stats["dead_forced"] == 2934 and len(labels) == n and set(labels) == {"D"}))

    ok = True
    for name, passed in cases:
        print("  [{}] {}".format("PASS" if passed else "FAIL", name))
        ok &= passed
    return ok


# -- part 2: synthetic match ---------------------------------------------------------------------

def synthetic_match(seed=42, half_frames=15000, n_halves=2):
    """Fake match. Returns (tracking, truth letters, halves). Truth has X outside the halves."""
    rng = np.random.RandomState(seed)
    gap = 500
    n = n_halves * half_frames + (n_halves + 1) * gap
    halves = [(gap + i * (half_frames + gap), gap + i * (half_frames + gap) + half_frames)
              for i in range(n_halves)]
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)

    # players: smooth random walks (normalised coordinates)
    vel = rng.normal(0, 0.0006, (n, 22, 2))
    for i in range(1, n):
        vel[i] = 0.95 * vel[i - 1] + 0.05 * vel[i]
    xy = np.clip(rng.uniform(0.1, 0.9, (1, 22, 2)) + np.cumsum(vel, axis=0), 0.02, 0.98)

    ball = np.full((n, 3), np.nan)
    truth = np.full(n, "X", dtype="<U1")
    jitter = 0.3 / np.array(PITCH)                               # 0.3 m of ball noise
    for start, stop in halves:
        f, team_now = start, rng.randint(2)
        while f < stop:
            if rng.rand() < 0.12:                                # stoppage: ball dead and unseen
                length = int(rng.uniform(8, 30) * FPS)
                truth[f:f + length] = "D"
                f += length
                team_now = rng.randint(2)
                continue
            length = max(2 * FPS, int(rng.exponential(10) * FPS))
            end = min(f + length, stop)
            truth[f:end] = "HA"[team_now]
            mates = np.flatnonzero(team == team_now)
            carrier, g = rng.choice(mates), f
            while g < end:                                       # dribble, then pass
                hold = min(end - g, int(rng.uniform(1, 4) * FPS))
                ball[g:g + hold, :2] = xy[g:g + hold, carrier] + rng.normal(0, 1, (hold, 2)) * jitter
                g += hold
                if g >= end:
                    break
                nxt = rng.choice(mates[mates != carrier])
                flight = min(end - g, int(rng.uniform(0.5, 1.5) * FPS))
                w = np.linspace(0, 1, flight + 2)[1:-1, None]
                ball[g:g + flight, :2] = (1 - w) * xy[g, carrier] + w * xy[g + flight, nxt]
                if rng.rand() < 0.3 and flight > 6:              # deflection off an opponent
                    opp = rng.choice(np.flatnonzero(team != team_now))
                    k = g + flight // 2
                    ball[k:k + 3, :2] = xy[k:k + 3, opp]
                g += flight
                carrier = nxt
            f = end
            team_now = 1 - team_now                              # next possession: other team
        # truth stoppages have no ball
        dead = (truth[start:stop] == "D")
        ball[start:stop][dead] = np.nan

    # degrade like our ATD: drop the ball in bursts (~41%), players ~5% at random
    unseen = np.zeros(n, dtype=bool)
    state = False
    for i in range(n):
        state = rng.rand() < (0.97 if state else 0.021)          # stationary share ~41%
        unseen[i] = state
    ball[unseen] = np.nan
    xy[rng.rand(n, 22) < 0.05] = np.nan
    return make_tracking(xy, team, ball, halves), truth, halves


def v0_style(tracking, window=50):
    """Nearest player + centred majority vote, as in starter/baseline.py."""
    sm_params = sm.default_params()
    obs, nearest = sm.frame_observations(tracking, sm_params)
    codes = np.where(np.isfinite(tracking.ball[:, 0]), nearest, -1)
    home = np.concatenate(([0], np.cumsum(codes == 0)))
    away = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    h, a = home[hi] - home[lo], away[hi] - away[lo]
    out = np.full(len(codes), "D", dtype="<U1")
    seen = (h + a) > 0
    out[seen] = np.where(h[seen] >= a[seen], "H", "A")
    return out


def spells(labels, halves):
    out = []
    for start, stop in halves:
        w = labels[start:stop]
        edges = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1, [len(w)]))
        out += [(start + edges[i], start + edges[i + 1], w[edges[i]]) for i in range(len(edges) - 1)
                if w[edges[i]] in ("H", "A")]
    return out


def spell_f1(truth, pred, halves):
    """Same idea as evaluation/possession_metrics.spell_scores (greedy one-to-one IoU match)."""
    g, p = spells(truth, halves), spells(pred, halves)
    pairs = []
    for i, (a, b, k) in enumerate(g):
        for j, (c, d, m) in enumerate(p):
            if k == m and c < b and a < d:
                pairs.append(((min(b, d) - max(a, c)) / float(max(b, d) - min(a, c)), i, j))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    f1s = []
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        used_g, used_p = set(), set()
        for iou, i, j in pairs:
            if iou < thr:
                break
            if i not in used_g and j not in used_p:
                used_g.add(i)
                used_p.add(j)
        tp = len(used_g)
        f1s.append(2.0 * tp / (len(g) + len(p)) if tp else 0.0)
    return float(np.mean(f1s)), len(g), len(p)


def part2():
    tracking, truth, halves = synthetic_match()
    v1, stats = sm.predict_v1(tracking, return_stats=True)
    v0 = v0_style(tracking)
    graded = truth != "X"
    print("  synthetic match: {} graded frames, ball unseen on {:.0f}% of them".format(
        int(graded.sum()), 100.0 * np.mean(~np.isfinite(tracking.ball[graded, 0]))))
    for name, pred in (("v0-style (nearest + smooth50)", v0), ("v1 state machine", v1)):
        f1, n_true, n_pred = spell_f1(truth, pred, halves)
        print("  {:<30} spellF1 {:.3f}  possessions {} (true {})  frame accuracy {:.3f}".format(
            name, f1, n_pred, n_true, float(np.mean(pred[graded] == truth[graded]))))
    for line in sm.describe_v1(v1, stats, halves):
        print("  " + line)
    return spell_f1(truth, v1, halves)[0] > spell_f1(truth, v0, halves)[0]


# -- part 3: v2 dead balls ----------------------------------------------------------------------

def dead_script(segments, tags=None):
    """Like script(), plus 'O' = ball out of the pitch, 'S' = ball still on the pitch next to the
    away player (a set piece he takes), 'K' = the away player kicks it (fast) after a restart,
    'P' = ball still in open space (nobody near), 'J' = the home player kicks it (fast).
    The 3rd ball channel is speed in m/s: 8 when the ball moves with a player, 0 when still."""
    t = script([(k if k in "HALNC" else ("H" if k == "J" else "L"), n) for k, n in segments])
    t.team_names = ("Barcelona", "Atletic Club")
    t.tags = tags
    f = 0
    for kind, length in segments:
        sl = slice(f, f + length)
        if kind in "HAL":
            t.ball[sl, 2] = 8.0
        elif kind == "O":
            t.ball[sl, :3] = (0.7, 1.05, 0.0)
        elif kind == "S":
            t.ball[sl, :3] = (0.7, 0.5, 0.0)
        elif kind == "K":
            t.ball[sl, :3] = (0.7, 0.5, 12.0)
        elif kind == "P":
            t.ball[sl, :3] = (0.5, 0.5, 0.0)
        elif kind == "J":
            t.ball[sl, :3] = (0.3, 0.5, 12.0)
        f += length
    return t


def part3():
    cases = []
    v2 = sm.predict_v2

    t = dead_script([("H", 100), ("N", 150), ("H", 100)])
    lab1, lab2 = sm.predict_v1(t), v2(t)
    cases.append(("ball missing 6 s in play: v1 calls D, v2 does not",
                  (lab1 == "D").sum() == 50 and set(lab2) == {"H"}))

    t = dead_script([("H", 100), ("O", 50), ("S", 50), ("K", 40), ("A", 60)])
    lab, st = v2(t, return_stats=True)
    d = np.flatnonzero(lab == "D")
    cases.append(("ball out then still -> D from leaving the pitch to the kick",
                  len(d) == 100 and d[0] == 100 and d[-1] == 199 and lab[200] == "A"))
    cases.append(("restart team from the nearest player (no tag) seeds the machine",
                  st["restart_team_used"] == 2 and set(lab[200:]) == {"A"}))   # kick-off + restart

    t = dead_script([("H", 100), ("P", 25), ("H", 100)])
    cases.append(("ball still for 1 s is not a stoppage", set(v2(t)) == {"H"}))

    tag = [{"code": "SET PIECES", "start_frame": 160, "end_frame": 230,
            "tags": {"Team": "Barcelona", "Type": "Free Kick"}}]
    t = dead_script([("A", 100), ("N", 100), ("J", 40), ("A", 60)], tags=tag)
    lab = v2(t)
    cases.append(("Smart Tagging set piece: D before the kick, tagged team restarts",
                  lab[99] == "A" and lab[150] == "D" and set(lab[200:240]) == {"H"}))

    n = sm.BAR_ATH_FRAMES
    t = make_tracking(np.full((n, 2, 2), np.nan), [0, 1], np.full((n, 3), np.nan), halves=None)
    t.halves, t.tags = None, []
    labels, stats = v2(t, return_stats=True)
    cases.append(("BAR-ATH layout under v2: pre-kickoff D is 2,934 frames, length 199,319",
                  stats["dead_forced"] == 2934 and len(labels) == n and labels[113741] == "D"))

    t = dead_script([("H", 100), ("N", 150), ("H", 100), ("N", 300), ("A", 100)])
    lab = v2(t, dead_mode="hybrid")
    d = np.flatnonzero(lab == "D")
    cases.append(("v2.1: 6 s ball gap stays play, 12 s gap becomes D (minus 2 s margins)",
                  set(lab[:350]) == {"H"} and len(d) == 200 and d[0] == 400 and lab[-1] == "A"))

    # v2.2: speed from positions. Home player dribbles (ball moves 3 m/s), ball goes out, is
    # carried back unseen, lies still, is kicked away by the away player, who keeps it.
    t = dead_script([("H", 100), ("O", 50), ("N", 10), ("S", 40), ("K", 40), ("A", 60)])
    t.ball[:, 2] = np.nan                                 # channel unusable: v2.2 must not need it
    run = np.linspace(0.20, 0.20 + 3.0 * 4 / 105.3, 100)  # 3 m/s for 4 s
    t.ball[:100, 0] = run
    t.players_xy[:100, 0, 0] = run
    t.ball[200:240, 0] = np.linspace(0.70, 0.78, 40)      # kicked: ~8.4 m in 1.6 s = 5 m/s
    t.ball[200:240, 1] = 0.5
    t.players_xy[200:240, 1, 0] = t.ball[200:240, 0]
    t.ball[240:, :2] = t.ball[239, :2] + np.linspace(0, 0.06, 60)[:, None] * [1, 0]
    t.players_xy[240:, 1] = t.ball[240:, :2]              # away player dribbles on
    lab = sm.predict_v2(t, dead_mode="hybrid", speed_source="positions")
    d = np.flatnonzero(lab == "D")
    cases.append(("v2.2: still/kick from positions, D from out until <= 3 frames after the kick",
                  len(d) and d[0] == 100 and 199 <= d[-1] <= 202 and lab[-1] == "A"
                  and set(lab[:100]) == {"H"}))

    from pipeline.dead_ball import speed_channel_check
    t = dead_script([("H", 100), ("A", 100)])
    t.ball[:, 0] = np.linspace(0.2, 0.8, 200)             # ball crosses the pitch in 8 s
    t.ball[:, 2] = 0.6 * 105.3 / 8.0                      # = its true speed
    good = speed_channel_check(t)
    t.ball[:, 2] = 0.5                                    # a channel that is not speed
    bad = speed_channel_check(t)
    cases.append(("speed check: ~100% when channel 3 is speed, ~0% when it is not",
                  good > 95 and bad < 5))

    ok = True
    for name, passed in cases:
        print("  [{}] {}".format("PASS" if passed else "FAIL", name))
        ok &= passed
    return ok


if __name__ == "__main__":
    print("Part 1: rule-by-rule cases")
    ok1 = part1()
    print("\nPart 2: synthetic match (smoke test, not for tuning)")
    ok2 = part2()
    print("\nPart 3: v2 dead balls")
    ok3 = part3()
    ok = ok1 and ok2 and ok3
    print("\n{}".format("ALL GOOD" if ok else "SOMETHING FAILED - paste this output to Claude"))
    sys.exit(0 if ok else 1)
