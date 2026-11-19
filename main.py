"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts each PS5 trophy's Exophase earn rate at the case's target as five
quantiles (0.05/0.25/0.5/0.75/0.95). Python 3.13, standard library only.

Model
-----
Exophase's page is a snapshot of two moving counts: the players it tracks and
the players who have earned each trophy. A rate is their ratio, so the model
forecasts the two counts separately and divides (the "decomposition"):

  * players_tracked: log-linear in log-time (ln(1 + hours since release)),
    fitted on the distinct page snapshots of the last 24 h (consecutive
    identical reads are one stale page, not new data).
  * earners per trophy: log(earners + 1) log-linear in log-time, per trophy,
    slope shrunk toward its tier's mean slope. Earners never decrease, so
    the series is smoothed with isotonic regression (PAVA) before fitting.
  * rate median (decomposition) = 100 * earners_target / players_target.

Two stabilisers keep the 4.5-day extrapolation honest. First, both fits use
only the last 24 h: launch-night growth is ancient history by the deadline
and must not drive the forecast. Second, extrapolated growth is capped
(players at most 5x, earners at most 20x over the horizon) and the
decomposition median is blended with the direct fit, so no single fit can
run away. The backtests favour near-linear extrapolation (damping scale 10
in log-time units, i.e. almost no damping): launch-week trends -- smooth
adoption curves and a steadily growing player base -- persist over 4.5
days, and damping them under-extrapolates badly (coverage 0.77 vs 0.96
undamped on 100 synthetic final-like cases).

The decomposition median is blended (in logit space) with a direct fit of
logit(rate) on log-time over the same window (the house-curve shape). The
blend hedges the decomposition's two-extrapolation error against the direct
fit's dilution blindness, and the disagreement between the two widens the
spread.

When the history is flat (no refresh seen yet, as on launch night) there is
no trend to fit, so the model falls back on launch priors calibrated on the
real launch-night refresh (638 -> ~3,900 tracked players in ~5 hours, every
rate but the 100% one falling): the player base grows as a decaying
exponential in real time, and each trophy's earners grow as a fraction of
the players' log-growth, the fraction rising with how many players already
hold the trophy (early-game trophies are earned fast, grinds barely move).
The spread is widened a lot, because an unseen refresh can move every rate
at once.

Spread: per-trophy logit spread combining the earners-fit residual, the
players-fit residual, a random-walk term growing with sqrt(horizon), a
refresh-jump term from the jumps actually seen in the history (or a wide
prior when none was seen), and the blend disagreement. Quantiles are the
median shifted by the normal z-scores times that spread.

Robustness: every trophy is forecast inside its own try/except and falls
back to persistence (last rate, house-style spread) on any failure; a
catastrophic case-level failure falls back to persistence for all trophies.
Output is always valid: one row per trophy, five finite non-decreasing
numbers in [0, 100], well under the 65,536-byte limit.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- tunable parameters (defaults chosen by synthetic + live-snapshot backtests)
RW_SIGMA = 0.25            # random-walk logit sigma per sqrt(day)
SPREAD_FLOOR = 0.30        # minimum per-trophy logit spread
SPREAD_CAP = 2.5           # maximum per-trophy logit spread
SHRINK_K = 3.0             # tier-shrinkage strength for earners slopes
FIT_WINDOW_HOURS = 24.0    # fits use only the most recent reads
MIN_FIT_POINTS = 3         # points needed for a trend fit, else persistence/prior
DAMP_TAU_X = 10.0          # log-time scale over which slopes saturate;
                           # 10 ~= no damping (backtests: launch-week trends
                           # persist; damping under-extrapolates)
BLEND_DECOMP = 0.5         # weight of the decomposition median (rest: direct fit)
PLAYERS_GROWTH_CAP = 5.0   # max players multiplier over the horizon
EARNERS_GROWTH_CAP = 20.0  # max earners multiplier over the horizon
FLAT_REFRESH_SPREAD = 0.8  # extra logit spread when no refresh has been seen
# flat-history launch prior: players log-growth rate r(t) = R0 * exp(-t/TAU)
PRIOR_R0 = 0.234           # per-hour log-growth rate at release
PRIOR_TAU_H = 15.0         # decay timescale of the launch influx (hours)
PRIOR_EARNERS_B = 0.7      # max earners/players log-growth ratio (flat prior)
PRIOR_EARNERS_EXP = 1.5    # how fast that ratio falls with the trophy's rate


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (matches JavaScript Math.round)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    """Parse an ISO-8601 UTC timestamp to epoch seconds (None on failure)."""
    if not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def parse_release(game):
    """Release instant (epoch seconds) from the case's game block."""
    if isinstance(game, dict):
        raw = game.get("release_date")
        if isinstance(raw, str):
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(raw[:19], fmt)
                    return dt.replace(tzinfo=timezone.utc).timestamp()
                except ValueError:
                    continue
    # GTA VI released 2026-11-19 (UTC); the competition's own default.
    return datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp()


def log_time(t, release):
    """ln(1 + hours since release); defined even slightly before release."""
    hours = (t - release) / 3600.0
    return math.log1p(max(hours, -0.99))


def damp(dx, tau=DAMP_TAU_X):
    """Saturating extrapolation distance: slope * damp(dx) instead of
    slope * dx, so a local trend cannot compound forever."""
    if dx <= 0:
        return dx
    return tau * (1.0 - math.exp(-dx / tau))


def wls(xs, ys, ws=None):
    """Weighted least squares for y = a + b*x. Returns (a, b, resid_std)."""
    if ws is None:
        ws = [1.0] * len(xs)
    sw = sum(ws)
    if sw <= 0 or len(xs) == 0:
        return 0.0, 0.0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        b = 0.0
    else:
        sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
        b = sxy / sxx
    a = my - b * mx
    if len(xs) > 2:
        var = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
        resid = math.sqrt(max(var, 0.0))
    else:
        resid = 0.0
    return a, b, resid


def pava(values):
    """Pool-adjacent-violators: non-decreasing least-squares fit (weights 1)."""
    n = len(values)
    if n <= 1:
        return list(values)
    out = []
    for v in values:
        out.append([float(v), 1.0])
        while len(out) >= 2 and out[-2][0] > out[-1][0]:
            v2, w2 = out.pop()
            v1, w1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    res = []
    for v, w in out:
        res.extend([v] * int(round(w)))
    return res[:n]


def dedup_reads(history):
    """Collapse consecutive identical reads into snapshots.

    Exophase refreshes on its own schedule; between refreshes the page is
    byte-identical, so a run of identical reads is one observation. Keep the
    first-seen time of each distinct page state (the values are as of the
    last refresh before it).
    Returns a list of (time, players, rates) and the first read time.
    """
    snaps = []
    first_t = None
    for read in history:
        try:
            players = read.get("players_tracked")
            rates = read.get("rates")
            if not isinstance(players, (int, float)) or players <= 0:
                continue
            if not isinstance(rates, list):
                continue
            t = parse_time(read.get("at")) or parse_time(read.get("slot"))
            if t is None:
                continue
            if first_t is None:
                first_t = t
            key = (players, tuple(rates))
            if snaps and snaps[-1][0] == key:
                continue
            snaps.append((key, t, float(players), list(rates)))
        except (TypeError, AttributeError, ValueError):
            continue
    return [(t, p, r) for _, t, p, r in snaps], first_t


def persistence_row(rate, horizon_days):
    """House-style fallback row: last rate, random-walk spread."""
    spread = 0.5 * math.sqrt(max(horizon_days, 0.0))
    lg = logit(rate)
    return [round4(clamp(inv_logit(lg + z * spread), 0.0, 100.0)) for z in Z]


def forecast_case(case, damp_tau=DAMP_TAU_X, blend_decomp=BLEND_DECOMP,
                  rw_sigma=RW_SIGMA, spread_floor=SPREAD_FLOOR):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_days = max(float(case.get("horizon_minutes") or 0) / 1440.0, 0.0)
    target_t = parse_time(case.get("target"))
    at_t = parse_time(case.get("at"))
    release = parse_release(case.get("game"))
    n = len(trophies)
    if n == 0:
        return {"trophies": []}

    snaps, first_t = dedup_reads(history)
    last_rates = list(snaps[-1][2]) if snaps else [None] * n
    last_players = snaps[-1][1] if snaps else None

    def fallback(i):
        r = last_rates[i]
        if r is None:
            r = 1.0
        return persistence_row(r, horizon_days)

    if not snaps or last_players is None:
        return {"trophies": [{"id": t["id"], "q": fallback(i)}
                            for i, t in enumerate(trophies)]}

    if target_t is None:
        target_t = (at_t or snaps[-1][0]) + horizon_days * 86400.0
    x_target = log_time(target_t, release)
    t_last = snaps[-1][0]
    t_anchor = first_t if first_t is not None else t_last
    flat = len(snaps) < 2

    # ---- fit window: the most recent reads (launch dynamics go stale fast)
    window_start = t_last - FIT_WINDOW_HOURS * 3600.0
    win = [s for s in snaps if s[0] >= window_start]
    if len(win) < MIN_FIT_POINTS:
        win = snaps
    win_x = [log_time(t, release) for t, _, _ in win]
    win_t = [t for t, _, _ in win]
    dx_target = x_target - win_x[-1]

    # ---- players model (trend branch)
    play_b, play_resid = 0.0, 0.0
    players_target = last_players
    if not flat and len(win) >= 2:
        _a, play_b, play_resid = wls(
            win_x, [math.log(p) for _, p, _ in win])
        play_b = min(max(play_b, 0.0), 2.0)
        mult = math.exp(play_b * damp(dx_target, damp_tau))
        players_target = last_players * min(mult, PLAYERS_GROWTH_CAP)

    # ---- flat-history launch prior for players (decaying real-time growth)
    if flat:
        h_anchor = max((t_anchor - release) / 3600.0, -0.99)
        h_target = (target_t - release) / 3600.0
        ln_mult = (PRIOR_R0 * PRIOR_TAU_H
                   * (math.exp(-h_anchor / PRIOR_TAU_H)
                      - math.exp(-h_target / PRIOR_TAU_H)))
        ln_mult = max(ln_mult, 0.0)
        players_target = last_players * math.exp(ln_mult)

    # ---- earners series per trophy (PAVA-smoothed, non-decreasing)
    earners = []  # per trophy: list of (x, t, earners) over window snapshots
    for i in range(n):
        series = []
        for (t, p, rates), x in zip(win, win_x):
            r = rates[i] if i < len(rates) else None
            if r is None:
                continue
            series.append((x, t, round(r / 100.0 * p)))
        if len(series) >= 2:
            vals = pava([e for _, _, e in series])
            series = [(x, t, v) for (x, t, _), v in zip(series, vals)]
        earners.append(series)

    # ---- per-trophy earners fits (trend branch), tier-shrunk slopes
    tiers = [t.get("tier", "bronze") for t in trophies]
    fits = []  # (slope, resid, n_points) or None
    for i in range(n):
        s = earners[i]
        if len(s) >= MIN_FIT_POINTS and not flat:
            _a, b, resid = wls([x for x, _, _ in s],
                               [math.log1p(e) for _, _, e in s])
            fits.append((max(b, 0.0), resid, len(s)))
        else:
            fits.append(None)

    def tier_mean(values):
        vals = [(v, w) for v, w in values if v is not None]
        if not vals:
            return None
        return sum(v * w for v, w in vals) / sum(w for _, w in vals)

    tier_slopes = {}
    tier_resids = {}
    for tr in set(tiers):
        own = [(fits[i][0], fits[i][2]) for i in range(n)
               if tiers[i] == tr and fits[i] is not None]
        allv = [(fits[i][0], fits[i][2]) for i in range(n)
                if fits[i] is not None]
        tier_slopes[tr] = tier_mean(own) if len(own) >= 2 else tier_mean(allv)
        if tier_slopes[tr] is None:
            tier_slopes[tr] = 0.0
        rs = sorted(fits[i][1] for i in range(n)
                    if tiers[i] == tr and fits[i] is not None)
        tier_resids[tr] = rs[len(rs) // 2] if rs else 0.15

    # ---- direct rate fits (trend branch): logit(rate) vs log-time
    rate_fits = []  # (slope, resid) or None
    for i in range(n):
        pts = []
        for (t, p, rates), x in zip(win, win_x):
            r = rates[i] if i < len(rates) else None
            if r is not None:
                pts.append((x, logit(r)))
        if len(pts) >= MIN_FIT_POINTS and not flat:
            _a, b, resid = wls([x for x, _ in pts], [y for _, y in pts])
            rate_fits.append((b, resid))
        else:
            rate_fits.append(None)

    # ---- refresh-jump scale from the history actually seen
    jumps = []
    for j in range(1, len(snaps)):
        prev, cur = snaps[j - 1][2], snaps[j][2]
        for i in range(n):
            a = prev[i] if i < len(prev) else None
            b = cur[i] if i < len(cur) else None
            if a is None or b is None:
                continue
            jumps.append(abs(logit(b) - logit(a)))
    jumps.sort()
    if jumps:
        jump_scale = jumps[int(0.75 * (len(jumps) - 1))]
        refresh_spread = min(0.5 * jump_scale, 0.3)
    else:
        refresh_spread = FLAT_REFRESH_SPREAD

    # ---- flat-prior earners slopes (rate-relative, early trophies grow fast)
    ref_rate = 1.0
    ln_players_mult = 0.0
    if flat:
        # the fastest trophy defines the scale: its earners track the
        # player influx most closely; slower trophies scale down from it
        rates_below_100 = [r for r in last_rates
                           if r is not None and r < 100.0]
        if rates_below_100:
            ref_rate = max(max(rates_below_100), 0.1)
        ln_players_mult = math.log(max(players_target / last_players, 1.0))

    rows = []
    for i, trophy in enumerate(trophies):
        try:
            last_r = last_rates[i]
            last_e = earners[i][-1][2] if earners[i] else 0.0
            if last_r is None:
                rows.append({"id": trophy["id"], "q": fallback(i)})
                continue

            # --- decomposition median
            f = fits[i]
            if f is not None:
                b_raw, resid, npts = f
                b = (npts * b_raw + SHRINK_K * tier_slopes[tiers[i]]) / (npts + SHRINK_K)
                b = max(b, 0.0)
                mult = math.exp(b * damp(dx_target, damp_tau))
                e_target = last_e * min(mult, EARNERS_GROWTH_CAP)
                e_target = max(e_target, last_e)
                resid_i = ((npts * resid + SHRINK_K * tier_resids[tiers[i]])
                           / (npts + SHRINK_K))
            elif flat:
                if last_r >= 100.0:
                    # everyone holds it: earners track players one-for-one
                    e_target = last_e * (players_target / last_players)
                    resid_i = 0.05
                else:
                    b_i = PRIOR_EARNERS_B * (max(last_r, 0.0) / ref_rate) ** PRIOR_EARNERS_EXP
                    b_i = min(max(b_i, 0.0), 1.0)
                    e_target = last_e * math.exp(b_i * ln_players_mult)
                    e_target = max(e_target, last_e)
                    resid_i = 0.15
            else:
                e_target = last_e
                resid_i = 0.15

            rate_decomp = clamp(100.0 * e_target / players_target)

            # --- direct rate-fit median (only when a real fit exists; in
            # the flat branch there is no trend, so blending with
            # persistence would just halve the launch prior's signal)
            rf = rate_fits[i]
            if rf is not None:
                b_r, _resid_r = rf
                lg_direct = logit(last_r) + b_r * damp(dx_target, damp_tau)
                rate_direct = clamp(inv_logit(lg_direct))
                lg_med = (blend_decomp * logit(rate_decomp)
                          + (1.0 - blend_decomp) * logit(rate_direct))
                disagreement = abs(logit(rate_decomp) - logit(rate_direct))
            else:
                lg_med = logit(rate_decomp)
                disagreement = 0.0
            rate_med = clamp(inv_logit(lg_med))

            spread = math.sqrt(
                resid_i ** 2
                + play_resid ** 2
                + (rw_sigma * math.sqrt(horizon_days)) ** 2
                + refresh_spread ** 2
                + (0.5 * disagreement) ** 2
            )
            spread = min(max(spread, spread_floor), SPREAD_CAP)
            q = [round4(clamp(inv_logit(lg_med + z * spread), 0.0, 100.0))
                 for z in Z]
            for k in range(1, 5):
                if q[k] < q[k - 1]:
                    q[k] = q[k - 1]
            rows.append({"id": trophy["id"], "q": q})
        except (TypeError, ValueError, OverflowError, ZeroDivisionError, KeyError):
            rows.append({"id": trophy["id"], "q": fallback(i)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except (OSError, ValueError, KeyError, TypeError):
            forecast = {"trophies": []}
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
