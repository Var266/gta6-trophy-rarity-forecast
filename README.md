# GTA VI Trophy Rarity Forecaster

Entry for make.fun's `gta6-trophy-rarity-forecast-01`: forecast each PlayStation 5
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
many players already hold the trophy. The spread is widened a lot, because an
unseen refresh can move every rate at once.

Spread: per-trophy logit spread combining the earners-fit residual, the
players-fit residual, a random-walk term growing with sqrt(horizon), a
refresh-jump term from the jumps actually seen in the history, and the
disagreement between the decomposition and the direct fit.

## Validation

- **Format**: valid output on both practice cases, ~0.1 s runtime (cap 8 s).
- **Live out-of-sample test**: fed the frozen launch-night history and asked
  to forecast 10 h ahead, against the real Exophase snapshot crawled at
  12:37Z — pinball loss **0.19** vs persistence **1.38**, median forecast
  ratio 0.184 vs the live 0.201, 90% coverage 0.98.
- **Synthetic final-like backtest** (100 scenarios, cohort-based launch-week
  simulator with refresh steps and 2-decimal rounding): mean pinball loss
  **0.247** vs fitted-curve baseline **0.312** vs persistence **0.469**;
  beats persistence in 50/50 scenarios and the fitted curve in 47/50.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
