# Camp Nou-ral Network — Technical document

More Than a Hack 2027 · FC Barcelona vs Athletic Club (LaLiga 2024-25, matchday 2)
Repository: https://github.com/spookysimi/camp-nou-ral-network
Submitted version: **v1** (`LABELER = "v1"` in `pipeline/config.py`)

## 1. Approach

We label every frame of the tactical video as H, A or D with a rule-based, fully deterministic pipeline built on the automatic tracking (ATD). No model is trained.

Because the ranking metric (spellF1) scores whole possessions, our main design goal was **stable possessions**: one wrong switch splits a possession and creates a false one. We therefore replaced the baseline's per-frame "nearest player + majority vote" with a **"sticky" state machine**: the team in possession keeps the ball until the other team *proves* control.

Development path (all versions are in the repository and selectable in `pipeline/config.py`):

| Version | What | Status |
|---|---|---|
| v0 | Rebuild of the organisers' `baseline_smooth50` on our own loader; byte-identical to the official file. Used to verify loading and frame alignment. | Verified, superseded |
| **v1** | Sticky state machine (rules below) + v0's dead-ball rule | **Submitted** |
| v2 / v2.1 / v2.2 | Dead-ball detector from tracking and Smart Tagging, three variants | Tried, dropped (section 6) |

## 2. Assets used and dropped

| Asset | Use |
|---|---|
| Tactical ATD tracking (metadata XML + raw TXT) | **Used.** Our own numpy parser (`pipeline/load.py`). Row i = video frame i (`frame_id - 1`); alignment checked: 199,319 records, ids 1..199,319, no gaps or duplicates. Teams and the team of each of the 46 track slots come from the metadata (first team = home = H). |
| Ball 3rd channel (believed to be speed) | Not used in v1. It agreed with speed computed from ball movement on only 33% of frames, so we did not trust it (section 6). |
| Smart Tagging | **Used only as a yardstick, not as input to v1:** `checks/smart_tagging_agreement.py` measures how often a submission gives the ball to the tagged team during attacking phases. Used as input in the dropped v2 variants. |
| Published facts | First graded frame / first-half kick-off (11,185) and second-half kick-off (113,742), from DATA.md / FAQ.md. |
| Homography, videos, broadcast feed | Not used. |
| Public 5-minute ground truth | Used only to check the pipeline (v0 reproduced the official baseline score there). Never used to tune or choose between versions. |

## 3. Possession rules (v1)

All thresholds are in `pipeline/config.py`.

1. **Control.** On each frame, a team *controls* the ball when one of its players is within **1.0 m** of the ball (distances in metres, using the 105.3 × 68.0 m pitch) and no opponent is. Both teams within 1.0 m = a duel; nobody = loose ball; ball not tracked = unknown.
2. **A switch needs proof.** Possession passes to the other team only after it controls the ball on **5 frames** (0.2 s), with gaps of at most **10 frames** (unknown, loose or duel) between them. Any control by the team in possession cancels the challenge. A touch or a deflection therefore does not change possession.
3. **Backdating.** A confirmed switch starts at the first frame of the new team's control, not when the proof is complete, because late transitions are penalised.
4. **Loose balls** (passes, long balls, bounces) stay with the team in possession.
5. **Duels** stay with the team in possession.
6. **Dead ball (D).** The ball is labelled dead when the tracking loses it for at least **4 s (101 frames)**, minus 2 s at each end of that stretch.
7. **Second-half start.** Frames 110,808–113,741 (graded but before the published second-half kick-off) are D. Frames outside the halves are D (not graded).
8. **After a stoppage** the machine starts fresh: the first team to prove control gets the ball, backfilled to the end of the stoppage. If nobody proves control in a live stretch, it goes to the team most often nearest to the ball.

We never abstain (no X labels).

## 4. Missing and noisy tracking

- **Missing ball (~41% of graded frames).** Missing frames never change possession (rules 2, 4, 5). A challenge is not broken by missing frames, because with 41% of ball positions missing, strict "5 frames in a row" would almost never fire.
- **Missing or swapped players.** Slots are track fragments, not people, so we only use each slot's team. Missing players are ignored in distance calculations.
- **Jitter.** The proof requirement (rule 2) is what removes flicker. On the full match, v1 gives 188 / 180 possessions per half (v0: ~300), with 7 / 1 shorter than 1 s (v0: ~60).

## 5. Training data

None. No labelled data was used; all rules and thresholds are hand-set priors. No hand labels on this match, no human-made data about this match.

## 6. Experiments tried and dropped

Without access to labels for this match, every version was judged with **label-free checks agreed before running it**:
- D share 25–50% in both halves
- 150–300 possessions per half
- Smart Tagging agreement on attacking entries ≥ 69.0%
- D inside attacking entries below v1's 20.5%

| | v1 (submitted) | v2 | v2.1 | v2.2 |
|---|---|---|---|---|
| What | missing ball 4 s+ = D | Dead-ball detector: ball out of pitch, ball still 2 s+, Smart Tagging set pieces and goals; restart team starts with the ball | v2 + missing ball 8 s+ = D | v2.1 with ball speed computed from positions |
| Possessions per half | 188 / 180 | 172 / 168 | 182 / 183 | 189 / 192 |
| D share per half | 33.1 / 34.8% | 21.9 / 16.2% | 36.4 / 38.0% | 39.2 / 39.8% |
| Smart Tagging agreement | 71.0% | 75.2% | 72.5% | 72.8% |
| D inside attacking entries | 20.5% | 13.1% | 24.4% | 26.4% |

v2 removed false stoppages during attacks but missed many real ones (too little D overall). Adding any "missing ball = dead" rule brought D back to a plausible share but also put D inside attacks.

**Test bench with real labels.** We also compared v1 and v2.2 on Metrica's open sample game 2: possession labels derived from its events (329 possessions; a team keeps the ball through its own actions, possession switches on recoveries, D from ball out until the restart), with the ball removed during live play so it is visible on ~59% of frames, like our tracking. Scored with the organisers' `run_evaluation.py` (1,000 bootstrap draws, noise band ~0.09):

| Metrica test match | v1 | v2.2 |
|---|---|---|
| spellF1 | **0.509** | 0.490 |
| macroF1 | **0.754** | 0.742 |
| D F1 | **0.687** | 0.671 |

The test match agrees with the label-free checks: v2.2 is not better, so we kept v1. The code is in the repository (`pipeline/dead_ball.py`, `LABELER = "v2"`, `"v2.1"`, `"v2.2"`).

## 7. Limitations

- **Dead balls** are the weakest part: v1 only detects stoppages where the tracking loses the ball, so it misses stoppages where the ball stays visible, and it calls D when the ball is lost for 4 s+ during play (about 20% of frames inside attacking Smart Tagging phases).
- **No restart team.** After a stoppage, v1 waits for proof of control instead of knowing who restarts.
- **Thresholds are priors**, not tuned. The Metrica test bench (section 6) became available only at the end and was used for one comparison; tuning on it is the next step.
- The test match differs from ours: its ball is removed at random during live play, while our tracking loses the ball in bursts during duels and crowded boxes.
- The 1.0 m control radius gives a clear controlling team on only ~24% of ball-visible frames; a larger radius may be better.

## 8. Reproduction

```bash
git clone https://github.com/spookysimi/camp-nou-ral-network
cd camp-nou-ral-network
python3.11 -m venv .venv && source .venv/bin/activate      # Python 3.11.9 used
pip install -r requirements-team.txt                        # exact versions we used
# put the match data in data/tactical/ (or set MTAH_DATA_DIR), as described in DATA.md
python run.py
```

`run.py` parses the tracking (cached in `data/cache/`), labels every frame with the version set in `pipeline/config.py` (`LABELER = "v1"`), writes `output/Camp Nou-ral Network.csv`, validates it with the organisers' `possession_metrics.validate_submission`, and prints only counts and summaries. The pipeline is deterministic: no randomness and no seeds involved. Self-tests without match data: `python checks/test_state_machine.py`.

## 9. Team and tools

Simi (pipeline), Attila (test bench), Fede (dead-ball detector, reproducibility), Hugo (possession rules, documentation), Nathan (project lead, rules for restarts). Code was written with help from AI assistants (Claude; Gemini), which never saw match data: only code, error messages and summary counts were shared.
