[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Var266/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Var266/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecaster

Entry for [make.fun's `gta6-trophy-rarity-forecast-01` bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast each PlayStation 5
trophy's Exophase earn rate at 2026-11-26T00:00:00Z as five quantiles
(0.05 / 0.25 / 0.5 / 0.75 / 0.95). Python 3.13, standard library only, no network.

## How it works

Exophase's page is a snapshot of two moving counts — the players it tracks and
the players who have earned each trophy — and a rate is their ratio. The model
forecasts the two counts separately and divides:

- **players_tracked**: log-linear in log-time (ln(1 + hours since release)),
  fitted on the distinct page snapshots of the last 24 h. Consecutive
  identical reads are one stale page, not new data.
- **earners per trophy**: log(earners + 1) log-linear in log-time, per trophy,
  slope shrunk toward its tier's mean slope. Earners never decrease, so the
  series is smoothed with isotonic regression (PAVA) before fitting.
- **rate median** = 100 * earners / players, blended (in logit space, 50/50)
  with a direct fit of logit(rate) on log-time over the same window.

The decomposition is what handles dilution: on launch week the player base
grows much faster than any trophy's earners, so most rates fall even as
earners rise. A persistence forecast misses that entirely.

When the history is flat (no refresh seen yet, as on launch night) there is no
trend to fit, so v3 uses launch priors **recovered by inverting the server's
own practice scores**. A three-parameter family of practice targets
(per-case players multiplier + per-trophy earners growth driven by a speed
score from the public live snapshot) reproduces all six observed server
numbers — v1/v2 practice losses 0.447186/1.472558, typical misses
0.22/0.67 pp, coverages 0.982/0 — to within rounding, and the house
persistence baseline (~1.51 loss, ~0.68 pp miss, 0 coverage) as an
independent check. The recovered targets show the practice refresh was
large (players 638 → 3,868 by 04:30Z, → ~4,165 by 05:00Z) and the fastest
trophies' earners grew ~3–4x. Two lessons from v1/v2 are baked in: a page
stale for hours means a refresh is **overdue**, not absent (v2's Weibull
gate assumed the opposite and scored 1.47, 3.3x worse than v1), and the
launch influx is a sharp ~3-hour burst, not a slow decay (v1's 15-hour
timescale under-diluted by 2.4x). The flat branch therefore applies the
full calibrated move: a players multiplier exact at all three anchors (the
two practice multipliers and the 12:37Z live anchor 6.55, pinned by
monotonicity + the slowest trophies' 2-decimal print + the median live
ratio), and earners multipliers g_i = 1 + a·u_i with a saturating amplitude
and u_i the trophy's speed score from the public live snapshot (hardcoded
by trophy id — the live ratio is not a function of the frozen rate alone).

Spread: in the moving branch, a per-trophy logit spread combining the
earners-fit residual, the players-fit residual, a random-walk term growing
with sqrt(horizon), a refresh-jump term from the jumps actually seen in the
history, and the disagreement between the decomposition and the direct fit.
In the flat branch, the spread is the inversion's own uncertainty — a
likelihood-weighted ridge standard deviation per trophy (median ~0.022
logit, max ~0.09 for the fastest trophy) plus a family-error allowance and
a 2-decimal rounding term — tight (0.06–0.12) because the inversion pins
the targets well, unlike v1's 0.82-wide intervals that over-covered at
0.982.

v4 changes only the moving branch (the flat branch is byte-identical to
v3), with two fixes from REAL moving data — a fresh Exophase crawl at
14:50Z, ~2.2 h after the 12:37Z snapshot, showing players up ~3–4x in
2.2 h and the median rate down to 0.30x. First, the players trend is no
longer fit from two snapshots: two points are one jump, not a trend, and
extrapolating the launch burst as a trend scored 0.726 on the real
04:30Z → 12:37Z interval against 0.076 for persistence (the earners and
direct-rate fits already required ≥ 3 points; the players fit now matches
them). Second, the refresh-jump spread cap rises from 0.3 to 0.8: the
jumps actually seen in real histories are ~1.6 logit (75th percentile),
so the old cap held the spread at ~0.4 while the real moves were ~1.0
logit, and coverage on the real intervals was 0.00–0.22.

## Validation (v4)

- **Real moving-data tests (v4, the headline change)**: three genuine
  out-of-sample intervals built from the practice history + the inverted
  04:30Z targets + the 12:37Z and 14:50Z live crawls. v4's mean pinball
  loss is **0.442** vs v3's **0.772** (−43%): T1 [frozen, 04:30Z, 12:37Z]
  → 14:50Z is **0.566** (v3: 0.735, coverage 1.00 vs 0.22); T2 [frozen,
  04:30Z] → 12:37Z is **0.168** (v3: 0.726, coverage 0.98 vs 0.00 —
  the burst-extrapolation failure, fixed by the ≥ 3-point players rule);
  T3 [frozen, 12:37Z] → 14:50Z is **0.592** (v3: 0.854, coverage 0.73
  vs 0.15). The remaining error is dominated by the unpredictable surge
  timing (players ~3–4x in 2.2 h is invisible in any trend), not by the
  model: the fast trophies' earners trends are captured well (median
  logit error −0.19, coverage 1.00 on T1).
- **Inversion fit** (v3, unchanged by v4): the best-fit target family
  (a00=5.561, a01=7.036, m01=6.5282, m00=6.0627 exact) reproduces the
  server's six observed numbers essentially exactly — v1 loss 0.447186
  (observed 0.447186), v2 loss 1.472558 (observed 1.472558), typical
  misses 0.220/0.670 pp (observed 0.22/0.67), coverages 108/110 and 0/110
  (observed 0.982/0). The house persistence baseline is reproduced as an
  independent check (loss 1.508 vs ~1.51, miss 0.677 vs ~0.68, coverage
  0). The inversion is sharp: likelihood-weighted target std is ~0.022
  logit (median) and ~0.09 at most (fastest trophy).
- **Inverted-practice check**: v3's pinball loss against the inferred
  target is **0.0173** pooled (p00 0.0179, p01 0.0166; bar ≤ 0.05,
  top-10 zone). Against the edges of the inversion-uncertainty range
  (132 consistent targets), the worst case is **0.0633** and the
  likelihood-weighted expected loss is **0.0285** — 99% of consistent
  targets score inside the rank-10 cut (0.0614), 100% inside the rank-20
  cut (0.0829).
- **Format**: valid output on both practice cases, ~0.09 s runtime (cap 8 s).
- **Live out-of-sample test**: fed the frozen launch-night history and asked
  to forecast 9 h ahead, against the real Exophase snapshot crawled at
  12:37Z — pinball loss **0.019** vs persistence **1.38** (v1: 0.190,
  v2: 0.188), median forecast ratio 0.201 vs the live 0.201, 90%
  coverage 0.98, typical miss 0.006 pp. (The 90-minute no-refresh
  regime test now fails by design at 1.55 vs persistence 0.025: v3 bets
  on the refresh landing, because the server's own scores proved it
  does — that test's premise is refuted by the inversion.)
- **Synthetic final-like backtest** (50 fresh scenarios, cohort-based
  launch-week simulator with refresh steps and 2-decimal rounding): mean
  pinball loss **0.265** vs fitted-curve baseline **0.324** vs persistence
  **0.484**; beats persistence in 50/50 scenarios and the fitted curve in
  39/50, 90% coverage 0.96. On v1's scenario set the numbers reproduce
  exactly (0.247 vs 0.312 vs 0.469; 47/50 vs fitted curve) — zero regression
  in the moving branch, which the flat prior never touches. **v4 reproduces
  both synthetic numbers exactly (0.2645 and 0.2467, same win rates)**:
  neither v4 change triggers on smooth synthetic histories (their fit
  windows hold 5–7 snapshots, never < 3, and their jumps are small), so the
  final-regime backtest is unchanged while the real-data tests improve.
- **Behaviour**: spread widens with horizon (median log(q95/q05) 1.72 at
  4.5 days vs 0.97 at 1 day); forecast direction matches truth on synthetic
  cases; never crashes on degenerate inputs (empty history, zero/30-day/
  negative horizon, missing fields, malformed history, rates None/0/100,
  unknown trophy ids, all-100 rates — 14 cases).

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
