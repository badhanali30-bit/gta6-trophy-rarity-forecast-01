"""GTA VI trophy-rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts every GTA VI (PS5) trophy's earn rate on Exophase at a case's
target, as five quantiles (5/25/50/75/95%), from the house's reads of the
trophy page before the case's checkpoint.

Python 3.13, standard library only, no network. Run:

    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

Reads every ``<id>.in`` in the input directory, writes ``<id>.out`` to the
output directory: ``{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}``.

The model
---------
An earn rate is a ratio, ``rate = 100 * E / N``, where ``E`` is how many of
the ``N`` players Exophase tracks have earned the trophy. The two move for
different reasons, and the page's history records both (each read prints
the players tracked and every rate, so ``E = rate * N / 100`` per read):

* ``N`` only ever grows, and fast just after launch: every new player
  starts with zero trophies, so growth in ``N`` dilutes every rate.
* ``E`` grows as tracked players play: fast for early, easy trophies
  (chapter completions, first activities), slowly or not at all for late,
  grindy ones (250 stunts, 100% completion, the platinum).

Forecasting the ratio directly (a trend on the rate itself) confuses the
two. So this program fits each side separately, in log space against
``x = ln(1 + hours since 2026-11-19T00:00:00Z)`` -- log time, because
both launch-week player growth and trophy earning decelerate -- and
divides the forecasts:

* ``ln N``: one pooled least-squares slope over all reads (a single
  series, shared by every trophy).
* ``ln E``: one least-squares slope per trophy, shrunk toward its tier's
  median slope (platinum / gold / silver / bronze trophies behave alike:
  story trophies are earned early, grind trophies late). A trophy with
  few distinct reads is pulled most of the way to its tier; a trophy with
  a long, consistent history keeps its own slope.
* Both forecasts are *level-anchored* at the last read: the fitted slope
  is applied forward from the last observed count, not from the
  regression line's level. A line fitted through the mean undershoots
  the endpoint whenever growth accelerates in log time (as launch-week
  earning does), which would forecast a phantom fall; anchoring at the
  last read keeps a flat history forecasting exactly the last rate, and
  a rising trophy rising from where it actually is.
* The target is read off at a *damped* ``x``: only part of the way from
  the last read's ``x`` to the target's ``x``. A trend fitted over ~2 days
  must not be linearly extrapolated 4.5 days out at full strength; the
  damping (and a hard cap on the total logit move from the last read)
  keeps a steep launch-week slope from forecasting absurd rates.
* The median rate is ``100 * E* / N*``, clamped to [0.01, 99.99]. A
  trophy at 100% stays ~100%: its ``E`` series *is* the ``N`` series, so
  the ratio stays at the top. A trophy nobody has earned stays near 0:
  its ``E`` line starts from a half-earner prior and barely moves.

The spread (quantiles) is honest about three sources of error, combined
in logit space around the median:

* a random walk, ``0.5 * sqrt(horizon in days)`` (the house baseline's
  scale -- rates genuinely wander);
* the fit residual: how far the history's rates sat from the model's own
  fitted curve, per trophy (0 when the history is flat, larger when the
  reads actually moved);
* a refresh allowance of 0.5 logit: Exophase refreshes its counts on its
  own schedule, not continuously, so the page can jump between reads --
  launch night it sat unchanged for 4+ hours (638 players tracked) and
  then jumped to 3,868 in one read, moving every rate but the 100% one.
  A band that ignores refreshes scores ~0 coverage on exactly that kind
  of jump; the allowance keeps the 90% band wide enough to honestly
  cover one.

Quantiles use the standard normal's z at each level, converted back with
the logistic function, rounded to 4 decimals half-up (as the house
baseline rounds). Output is deterministic: cases are processed in file
name order, trophies in listed order, no randomness, no wall-clock or
hash-order dependence.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves the entry directory OFF sys.path; put it back
# first, so a helper module shipped beside this file would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Hours are measured from the start of release day, 2026-11-19T00:00:00Z,
# the origin the house baselines use for their log-time fits.
EPOCH = "2026-11-19T00:00:00Z"

# Random-walk spread, in logit units per square-root day.
SIGMA_RW = 0.5
# Refresh allowance: one Exophase refresh can move a rate's logit by this
# much (1 sigma). Launch night's first refresh moved most rates by ~1.8
# logit (a 6x jump in players tracked, 638 -> 3,868, in a single read);
# later refreshes are smaller as the player base grows, so 0.7 is a
# compromise: the 90% band reaches ~1.15 logit below the median, covering
# a ~3x dilution jump, without drowning the forecast in width.
SIGMA_REFRESH = 0.7
# How much of the log-time gap between the last read and the target the
# trend is allowed to cross (1.0 = full extrapolation, 0 = persistence).
DAMP = 0.7
# Hard cap on how far the median's logit may move from the last read.
MAX_LOGIT_MOVE = 2.0
# Shrinkage strength for a trophy's log-earners slope toward its tier
# median: weight on the tier slope is SHRINK_K / (n + SHRINK_K).
SHRINK_K = 2.0


def _parse_time_cached(text: str) -> float:
    """ISO-8601 UTC timestamp -> seconds since the Unix epoch.

    Handles the trailing ``Z`` and fractional seconds the house writes.
    """
    from datetime import datetime, timezone

    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    moment = datetime.fromisoformat(cleaned)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


EPOCH_SECONDS = _parse_time_cached(EPOCH)


def hours_since_epoch(at: str) -> float:
    """Hours from 2026-11-19T00:00:00Z to the given timestamp."""
    return (_parse_time_cached(at) - EPOCH_SECONDS) / 3600.0


def log_time(at: str) -> float:
    """``ln(1 + hours since the epoch)`` for a read timestamp."""
    return math.log1p(max(0.0, hours_since_epoch(at)))


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def logistic(lg: float) -> float:
    """Logit -> percent (0..100), overflow-safe."""
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as the house baseline rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares ``y = a + b*x``; returns (intercept, slope).

    With fewer than two distinct x values the line is flat at the mean.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return mean_y, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return mean_y - slope * mean_x, slope


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def distinct_reads(history: list[dict]) -> list[dict]:
    """Reads with duplicate (time, players, rates) vectors removed.

    Launch night the page sat unchanged for hours; keeping every copy
    would overweight one frozen snapshot in the fits.
    """
    seen: set[tuple] = set()
    kept: list[dict] = []
    for read in history:
        key = (read.get("at"), read.get("players_tracked"),
               tuple(read.get("rates") or ()))
        if key in seen:
            continue
        seen.add(key)
        kept.append(read)
    return kept


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0

    reads = distinct_reads(history)
    # Log-time of each distinct read, and of the checkpoint / target.
    xs = [log_time(r["at"]) for r in reads]
    x_last = xs[-1]
    x_target = log_time(case["target"])
    # Damped target: part-way from the last read toward the target.
    x_eff = x_last + DAMP * max(0.0, x_target - x_last)

    # --- players tracked: one pooled log-line over all reads. ----------
    n_points: list[tuple[float, float]] = []
    for read, x in zip(reads, xs):
        players = read.get("players_tracked")
        if players and players > 0:
            n_points.append((x, math.log(float(players))))
    last_n = float(reads[-1]["players_tracked"])
    if len(n_points) >= 2:
        _n_a, n_b = ols([p[0] for p in n_points], [p[1] for p in n_points])
        # Level-anchored: grow forward from the last read's count.
        log_n_star = math.log(last_n) + n_b * (x_eff - x_last)
        # Never forecast fewer tracked players than the last read saw.
        log_n_star = max(log_n_star, math.log(last_n))
    else:
        log_n_star = math.log(last_n)
    n_star = math.exp(log_n_star)

    # --- per-trophy log-earners lines, then tier shrinkage. ------------
    # First pass: each trophy's own slope, from its non-null reads.
    own: list[dict] = []
    for i, trophy in enumerate(trophies):
        points: list[tuple[float, float]] = []
        for read, x in zip(reads, xs):
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            players = read.get("players_tracked") or 0
            earners = max(0.0, float(rates[i]) * float(players) / 100.0)
            # No +0.5 fudge when earners > 0: adding half an earner would
            # inflate small counts (4 earners -> 4.5, +12%) and bias every
            # median upward. The half-earner prior applies only to a
            # trophy nobody has earned yet, where log(0) is undefined.
            points.append((x, math.log(earners if earners > 0 else 0.5)))
        if len(points) >= 2:
            a, b = ols([p[0] for p in points], [p[1] for p in points])
        elif len(points) == 1:
            a, b = points[0][1], 0.0
        else:
            a, b = math.log(0.5), 0.0
        own.append({"a": a, "b": b, "n": len(points),
                    "tier": trophy.get("tier", ""), "points": points})

    # Tier median slopes (a tier's trophies are earned alike).
    tier_slopes: dict[str, list[float]] = {}
    for entry in own:
        tier_slopes.setdefault(entry["tier"], []).append(entry["b"])
    tier_median = {t: median(v) for t, v in tier_slopes.items()}

    # --- median + spread per trophy. -----------------------------------
    last = reads[-1]
    last_rates = last.get("rates") or []
    rows = []
    for i, trophy in enumerate(trophies):
        entry = own[i]
        # Shrink the slope toward the tier median; few reads -> tier wins.
        w_tier = SHRINK_K / (entry["n"] + SHRINK_K)
        slope = (1.0 - w_tier) * entry["b"] + w_tier * tier_median.get(entry["tier"], 0.0)
        # Intercept of the shrunk-slope line, through the points' mean:
        # used only for the fit residual below, not for the forecast.
        if entry["points"]:
            mean_x = sum(p[0] for p in entry["points"]) / len(entry["points"])
            mean_y = sum(p[1] for p in entry["points"]) / len(entry["points"])
            intercept = mean_y - slope * mean_x
        else:
            intercept = entry["a"]
        # Level-anchored: grow forward from the last read's earners.
        if entry["points"]:
            x_last_e, y_last_e = entry["points"][-1]
            log_e_star = y_last_e + slope * (x_eff - x_last_e)
            # Earners never decrease: floor at the last read's earners.
            log_e_star = max(log_e_star, y_last_e)
        else:
            log_e_star = math.log(0.5)
        # Earners cannot exceed the players tracked.
        log_e_star = min(log_e_star, log_n_star)
        rate_star = 100.0 * math.exp(log_e_star) / n_star

        # Median, capped in how far it may move from the last read.
        last_rate = last_rates[i] if i < len(last_rates) and last_rates[i] is not None else rate_star
        med_logit = logit(rate_star)
        last_logit = logit(last_rate)
        move = max(-MAX_LOGIT_MOVE, min(MAX_LOGIT_MOVE, med_logit - last_logit))
        med_logit = last_logit + move

        # Fit residual: how far the history's log-earners sat from the
        # model's own fitted line (0 when the history is flat). Measured
        # in log-earner space, not logit space: near 0% or 100% a tiny
        # rate wobble is a huge logit move, which would blow the spread
        # up for the most certain trophies (a 100% trophy's band once
        # reached down to 61% on that mistake). Capped at 1 logit.
        residuals: list[float] = []
        for x, y in entry["points"]:
            residuals.append(y - (intercept + slope * x))
        if len(residuals) >= 2:
            mean_r = sum(residuals) / len(residuals)
            resid_sd = min(1.0, math.sqrt(
                sum((r - mean_r) ** 2 for r in residuals) / (len(residuals) - 1)))
        else:
            resid_sd = 0.0

        sigma = math.sqrt((SIGMA_RW * math.sqrt(horizon_days)) ** 2
                          + resid_sd ** 2 + SIGMA_REFRESH ** 2)
        quantiles = []
        for z in Z:
            value = logistic(med_logit + z * sigma)
            quantiles.append(round4(min(100.0, max(0.0, value))))
        # Rounding can only move a value by 0.00005; enforce monotonicity.
        for k in range(1, len(quantiles)):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})

    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
