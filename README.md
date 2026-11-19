# GTA VI Trophy Rarity Forecast

Forecasts every Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at
a case's target, as five quantiles (5/25/50/75/95%) per trophy.

Entry for [gta6-trophy-rarity-forecast-01](https://make.fun/c/gta6-trophy-rarity-forecast-01)
($250; final target 2026-11-26T00:00:00Z, closes 2026-11-21T11:30:00Z).

## How to run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `<id>.in` in
the input directory, writes `<id>.out` to the output directory:
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`.

## The model

An earn rate is a ratio, `rate = 100 * E / N`: `E` earners out of `N`
players Exophase tracks. The two move for different reasons — `N` only
grows (every new player starts with zero trophies, so growth in `N`
dilutes every rate), while `E` grows as players play (fast for early
story trophies, slow for grinds). So `main.py` fits each side separately
in log space against `ln(1 + hours since 2026-11-19T00:00:00Z)` and
divides the forecasts:

- `ln N`: one pooled least-squares slope over all reads.
- `ln E`: one least-squares slope per trophy, shrunk toward its tier's
  (platinum/gold/silver/bronze) median slope.
- Both forecasts are **level-anchored** at the last read (the slope is
  applied forward from the last observed count), and read off at a
  **damped** target (`x_last + 0.7 * (x_target - x_last)`), with a hard
  cap of 2.0 logit on the median's move from the last read — a trend
  fitted over ~2 days must not be extrapolated 4.5 days out at full
  strength.
- Rates are clamped to [0.01, 99.99]. A trophy at 100% stays ~100%
  (its `E` series *is* the `N` series); an unearned trophy stays near 0.

The spread is honest about three errors, combined in logit space:
random walk `0.5 * sqrt(horizon in days)`, the per-trophy fit residual
(measured in log-earner space, capped at 1.0), and a **0.7 logit
refresh allowance** — Exophase refreshes on its own schedule, and launch
night it sat unchanged for 4+ hours (638 players) then jumped to 3,868
in one read, moving every rate but the 100% one. A band that ignores
refreshes scores ~0 coverage on exactly that kind of jump.

See the full design notes in the docstring at the top of `main.py`.

## Validation (local, practice cases p00/p01)

- Both outputs valid: 55 rows each, ids 1–55, quantiles non-decreasing,
  2,686 bytes each (limit 65,536).
- Deterministic: two runs byte-identical (`diff` clean).
- Runtime ~0.05 s per run (limit 8 s per case).
- Practice histories are completely flat (all reads identical: 638
  players, rates unchanged), so medians equal the last read — as they
  should; no model can see the launch-night refresh coming from a flat
  history. What differs from the starter is the spread: median 90%
  band width 2.23 points vs the starter's 0.32 (≈7× wider).
- Against a worst-case proxy answer (refresh with earners frozen,
  rates ÷ 6.06): pinball loss 1.27 vs the starter's 1.69. The real
  answer will be kinder — new players do earn trophies — but the
  direction is the point: the wider band pays less when a refresh lands.
- Synthetic moving-history check (N 638→3,000, an early trophy's
  earners 10→900, a late trophy's 4→6, 5-day horizon): the early
  trophy's median rises 30% → 53.8%, the late trophy's falls
  0.20% → 0.15%, the 100% trophy stays ~99.99%, a flat trophy stays
  0.63%. The E/N decomposition moves medians the right way.
