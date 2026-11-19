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
trend to fit, so the model falls back on launch priors calibrated on the real
launch-night refresh (638 -> ~3,900 tracked players in ~5 hours): the player
base grows as a decaying exponential in real time, and each trophy's earners
grow as a fraction of the players' log-growth, the fraction rising with how
many players already hold the trophy.

The launch prior is horizon-aware: Exophase only moves when it refreshes, and
refreshes arrive roughly once per half day to a day, so over a 90-minute
horizon the page almost certainly does not move at all, while over several
days it certainly refreshes. The prior scales its move by the probability a
refresh lands inside the horizon, P(h) = 1 - exp(-(h/6.5)^3) (Weibull,
modelling the residual wait given the page is already stale): near
persistence at 90 minutes, nearly the full dilution drop at multi-day
horizons. The flat refresh-spread term is scaled by the same probability.
(v1 applied the full dilution drop at every horizon and scored 0.447 on the
practice gate, whose 90-minute targets barely move; v2 fixes exactly that.)

Spread: per-trophy logit spread combining the earners-fit residual, the
players-fit residual, a random-walk term growing with sqrt(horizon), a
refresh-jump term from the jumps actually seen in the history, and the
disagreement between the decomposition and the direct fit.

## Validation (v2)

- **Format**: valid output on both practice cases, ~0.07 s runtime (cap 8 s).
- **90-minute no-refresh regime** (the practice gate: frozen history,
  90-minute horizon, target = frozen rates): pinball loss **0.034** vs
  persistence **0.025** (1.37x, bar 1.5x); median forecast ratio 0.991.
  v1 scored 0.41 here (16-17x persistence) and 0.447 on the real gate.
- **Live out-of-sample test**: fed the frozen launch-night history and asked
  to forecast 9 h ahead, against the real Exophase snapshot crawled at
  12:37Z — pinball loss **0.188** vs persistence **1.38** (v1: 0.190),
  median forecast ratio 0.205 vs the live 0.201, 90% coverage 0.98.
- **Synthetic final-like backtest** (50 fresh scenarios, cohort-based
  launch-week simulator with refresh steps and 2-decimal rounding): mean
  pinball loss **0.265** vs fitted-curve baseline **0.324** vs persistence
  **0.484**; beats persistence in 50/50 scenarios and the fitted curve in
  39/50, 90% coverage 0.96. On v1's scenario set the numbers reproduce
  exactly (0.247 vs 0.312 vs 0.469; 47/50 vs fitted curve) — zero regression
  in the moving branch, which the flat prior never touches.
- **Behaviour**: spread widens with horizon (median log(q95/q05) 1.72 at
  4.5 days vs 0.97 at 1 day); forecast direction matches truth on synthetic
  cases; never crashes on degenerate inputs (empty history, zero horizon,
  missing fields, all-zero/all-100 rates).

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
