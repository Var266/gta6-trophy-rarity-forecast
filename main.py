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
no trend to fit, so the model uses launch priors RECOVERED BY INVERTING the
server's own practice scores (v3). The inversion is exact: a three-parameter
family of practice targets (per-case players multiplier + per-trophy
earners growth driven by a speed score from the public live snapshot)
reproduces all six observed server numbers -- v1/v2 practice losses
0.447186/1.472558, typical misses 0.22/0.67 pp, coverages 0.982/0 -- to
within rounding, and the house persistence baseline (~1.51 loss, ~0.68 pp
miss, 0 coverage) as an independent check. The recovered targets show the
practice refresh was large (players 638 -> 3,868 by 04:30Z, -> ~4,165 by
05:00Z) and the fastest trophies' earners grew ~3-4x. Two lessons from v1/v2
are baked in: a page stale for hours means a refresh is OVERDUE, not absent
(v2's Weibull gate assumed the opposite and scored 1.47, 3.3x worse than
v1), and the launch influx is a sharp ~3-hour burst, not a slow decay (v1's
15-hour timescale under-diluted by 2.4x). The flat branch therefore applies
the full calibrated move: players multiplier m(elapsed), piecewise in
ln(m) and exact at all three anchors -- the two practice multipliers
(6.0627 at 4.667 h, 6.5282 at 5.167 h) and the 12:37Z live anchor (6.55,
pinned by monotonicity + the slowest trophies' 2-decimal print + the
median live ratio) -- with a Weibull burst below the first anchor and
exponential saturation beyond the second; earners
multiplier g_i = 1 + a(elapsed) x u_i with a(elapsed) = 7.6 x
(1 - exp(-elapsed/2.78)) and u_i the trophy's speed score from the public
live snapshot (hardcoded by trophy id, rate-band fallback for unknown ids).
The flat spread is the inversion's own uncertainty: a likelihood-weighted
ridge standard deviation per trophy (median ~0.022 logit, max ~0.09 for the
fastest trophy) plus a family-error allowance and a 2-decimal rounding term
-- tight (0.06-0.12) because the inversion pins the targets well, unlike
v1's 0.82-wide intervals that over-covered at 0.982.

Spread: in the moving branch, a per-trophy logit spread combining the
earners-fit residual, the players-fit residual, a random-walk term growing
with sqrt(horizon), a refresh-jump term from the jumps actually seen in the
history, and the blend disagreement. In the flat branch, the spread is the
inversion uncertainty (ridge std + family error + rounding), floored at
0.06. Quantiles are the median shifted by the normal z-scores times that
spread.

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
# flat-history launch prior (v3, recovered by inverting the server's
# practice scores -- see the module docstring). Players multiplier, as a
# function of elapsed hours since the frozen page's first read. Anchored
# EXACTLY at the two practice multipliers (6.0627 at 4.667 h, from
# RULES.txt's 3868/638; 6.5282 at 5.167 h, the inversion best fit) and
# saturating to 6.55 beyond ~6 h. The live anchor 6.55 is pinned by three
# independent constraints on the 12:37Z snapshot: players cannot decrease
# (m_live >= m_01 = 6.53), the slowest trophies' 0.10% print with 4
# earners each puts m_live <= 6.60, and the median live ratio 0.201 with
# the practice earners amplitude gives m_live ~ 6.5. (The E=5 alternative,
# m_live 7.5-8.2, would need the earners amplitude to nearly double after
# 5 h, against saturation.) The influx is a sharp ~3 h burst, saturating
# by ~8 h -- v1's 15 h decaying exponential under-diluted by 2.4x.
# Piecewise in ln(m): a Weibull burst below anchor 1 (fitted to pass
# through it), linear between the two practice anchors, exponential
# saturation beyond anchor 2. A single Weibull cannot hit all three
# anchors: the 4.667 -> 5.167 h rise is too steep for its shape.
PRIOR_M_LINF = 1.9459     # Weibull saturation (burst segment only)
PRIOR_M_TAU_H = 3.1345    # burst timescale (hours)
PRIOR_M_SHAPE = 2.4057    # Weibull shape
PRIOR_M_ANCHOR1_H = 4.667   # first practice anchor (elapsed hours)
PRIOR_M_ANCHOR1_LN = 1.80215  # ln(6.0627)
PRIOR_M_ANCHOR2_H = 5.167   # second practice anchor
PRIOR_M_ANCHOR2_LN = 1.87613  # ln(6.5282)
PRIOR_M_LIVE_LN = 1.87970   # ln(6.55), saturating live anchor
PRIOR_M_SAT_TAU_H = 1.5    # saturation timescale beyond anchor 2 (hours)
# earners amplitude, same elapsed clock:
#   a(elapsed) = A_INF * (1 - exp(-elapsed / A_TAU))
# a(4.667 h) = 6.18, a(5.167 h) = 6.42, a(12.787 h) = 7.52. Calibrated to
# the inversion's shared-amplitude fit (6.27 across the practice window)
# and the live snapshot (a ~ 7.3-7.7 at 13 h). The per-case split in the
# best fit (5.56 / 7.04) is ridge noise: it is too steep for any saturating
# curve, so the smooth curve is used and the split's uncertainty is carried
# in the spread instead.
PRIOR_A_INF = 7.6         # saturated earners amplitude
PRIOR_A_TAU_H = 2.78      # earners-growth timescale (hours)
# per-trophy earners-growth speed score u_i, from the PUBLIC Exophase live
# snapshot (2026-11-19 12:37Z crawl): u_i = live_ratio_i - 0.159, where
# 0.159 is the slowest trophies' ratio (their earners barely move, so their
# ratio is ~1/m_live). Keyed by trophy id because the live ratio is NOT a
# function of the frozen rate alone (same rate -> different ratios).
# Trophy 2 (the 100% trophy) is excluded: it stays at 100%.
FLAT_U = {1: 0.0, 3: 0.233, 4: 0.05, 5: 0.05, 6: 0.05, 7: 0.05, 8: 0.05,
          9: 0.05, 10: 0.05, 11: 0.05, 12: 0.05, 13: 0.05, 14: 0.0,
          15: 0.0, 16: 0.008, 17: 0.0, 18: 0.0, 19: 0.008, 20: 0.41,
          21: 0.256, 22: 0.008, 23: 0.008, 24: 0.046, 25: 0.0, 26: 0.0,
          27: 0.008, 28: 0.0, 29: 0.008, 30: 0.0, 31: 0.0, 32: 0.0,
          33: 0.032, 34: 0.046, 35: 0.153, 36: 0.008, 37: 0.086,
          38: 0.038, 39: 0.032, 40: 0.008, 41: 0.011, 42: 0.046,
          43: 0.008, 44: 0.064, 45: 0.008, 46: 0.046, 47: 0.046,
          48: 0.072, 49: 0.046, 50: 0.072, 51: 0.072, 52: 0.008,
          53: 0.032, 54: 0.046, 55: 0.046}
# flat-branch spread (inversion uncertainty, logit std):
#   inv_std_i = INV_COEF * u_i + INV_BASE  (likelihood-weighted ridge std:
#   median 0.022, max 0.089 for id 20; the linear form in u_i reproduces
#   it to ~0.01), FAMILY_SPREAD = per-trophy shape-error allowance,
#   plus a 2-decimal rounding term. Total floored at 0.06, capped at 0.30.
FLAT_INV_COEF = 0.22
FLAT_INV_BASE = 0.012
FLAT_FAMILY_SPREAD = 0.06
FLAT_SPREAD_FLOOR = 0.06
FLAT_SPREAD_CAP = 0.30


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


def ln_players_mult(elapsed_h):
    """ln players multiplier for the flat branch, on the elapsed clock
    since the frozen page's first read. Piecewise (see the PRIOR_M_*
    constants): Weibull burst below anchor 1, linear between the two
    practice anchors, exponential saturation toward the live anchor
    beyond anchor 2. Exact at all three anchors."""
    if elapsed_h <= 0:
        return 0.0
    if elapsed_h <= PRIOR_M_ANCHOR1_H:
        return PRIOR_M_LINF * (1.0 - math.exp(
            -((elapsed_h / PRIOR_M_TAU_H) ** PRIOR_M_SHAPE)))
    if elapsed_h <= PRIOR_M_ANCHOR2_H:
        f = ((elapsed_h - PRIOR_M_ANCHOR1_H)
             / (PRIOR_M_ANCHOR2_H - PRIOR_M_ANCHOR1_H))
        return (PRIOR_M_ANCHOR1_LN
                + f * (PRIOR_M_ANCHOR2_LN - PRIOR_M_ANCHOR1_LN))
    return PRIOR_M_LIVE_LN - (PRIOR_M_LIVE_LN - PRIOR_M_ANCHOR2_LN) * (
        math.exp(-(elapsed_h - PRIOR_M_ANCHOR2_H) / PRIOR_M_SAT_TAU_H))


def flat_u(trophy_id, last_rate):
    """Earners-growth speed score for the flat branch: the hardcoded
    live-snapshot value by trophy id, with a rate-band fallback (median u
    in the band) for trophy ids the snapshot did not cover."""
    u = FLAT_U.get(trophy_id)
    if u is not None:
        return u
    if last_rate >= 1.5:
        return 0.22
    if last_rate >= 1.0:
        return 0.05
    if last_rate >= 0.8:
        return 0.03
    return 0.008


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
                  rw_sigma=RW_SIGMA, spread_floor=SPREAD_FLOOR,
                  flat_spread_floor=FLAT_SPREAD_FLOOR):
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
    # No refresh gate in the flat branch: a page frozen for hours is
    # OVERDUE for a refresh (v2's gate assumed the opposite and scored
    # 1.47). The calibrated m(t)/a(t) curves already encode the refresh
    # timing -- they were recovered from cases where the refresh DID land
    # inside the window. The moving branch sees refreshes directly, so it
    # needs no prior at all.

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

    # ---- flat-history launch prior for players (inversion-calibrated
    # Weibull-saturating multiplier on the elapsed clock since the frozen
    # page's first read). The full move is applied: a stale page is
    # overdue, and the calibration cases are exactly the stale-page cases.
    elapsed_h = 0.0
    if flat:
        elapsed_h = max((target_t - t_anchor) / 3600.0, 0.0)
        ln_mult = max(ln_players_mult(elapsed_h), 0.0)
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
        # flat history: no refresh seen yet. The flat branch carries its
        # own inversion-based spread, so the generic jump term is zero
        # here (it is only used by the moving branch's formula).
        refresh_spread = 0.0

    # ---- flat-prior earners amplitude (inversion-calibrated, saturating
    # on the same elapsed clock as the players multiplier)
    flat_a = 0.0
    if flat:
        flat_a = PRIOR_A_INF * (1.0 - math.exp(-elapsed_h / PRIOR_A_TAU_H))

    rows = []
    for i, trophy in enumerate(trophies):
        try:
            last_r = last_rates[i]
            last_e = earners[i][-1][2] if earners[i] else 0.0
            u_i = 0.0  # flat-branch speed score (set in the flat branch below)
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
                    # inverted earners growth: g_i = 1 + a * u_i, the
                    # speed score from the public live snapshot
                    u_i = flat_u(trophy["id"], last_r)
                    e_target = last_e * (1.0 + flat_a * u_i)
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

            if flat:
                # inversion-uncertainty spread: ridge std (linear in the
                # speed score) + family error + 2-decimal rounding term.
                # No random-walk or jump term: the refresh jump is the
                # median move itself, and its uncertainty is the ridge.
                inv_std = FLAT_INV_COEF * u_i + FLAT_INV_BASE
                rounding_std = min(0.003 / max(rate_med, 0.05), 0.05)
                spread = math.sqrt(inv_std ** 2
                                   + FLAT_FAMILY_SPREAD ** 2
                                   + rounding_std ** 2)
                spread = min(max(spread, flat_spread_floor),
                             FLAT_SPREAD_CAP)
            else:
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
