"""Synthetic time-series scenario library for the TimesFM POC.

Each scenario is a deterministic (seeded) generator producing a DataFrame
with a `ds` (date) column, a `y` (target) column, and optionally covariate
columns. Series are grouped into categories, several of which are
deliberately "hard to forecast" (very short history, abrupt level shifts,
overlapping seasonal cycles, intermittent demand, volatility regime change).

Each category has ~10 instances spanning daily/weekly/monthly/annual frequency
and varying scale, trend direction, and (where applicable) seasonality/noise, so
switching between series is a meaningful exploration rather than cosmetic
variation on one template.
"""

from __future__ import annotations

import dataclasses
import itertools
import string
from functools import lru_cache

import numpy as np
import pandas as pd

FREQ_LABELS = {"D": "Daily", "W": "Weekly", "MS": "Monthly", "YS": "Annual"}
FREQ_UNIT = {"D": "days", "W": "weeks", "MS": "months", "YS": "years"}
FREQ_YEARLY_PERIOD = {"D": 365.25, "W": 52.18, "MS": 12.0}  # not applicable to YS -- a year IS the base unit
# W is 5 years (not 3): classical ETS/ARIMA need several full cycles to reliably
# estimate a 52-week seasonal component. At 3 years (~2.6 cycles after a holdout
# carve-out) AutoETS's own AIC-based model selection correctly declines to fit a
# seasonal term at all -- not a bug, just insufficient data for that period length.
# 5 years (~4-4.6 cycles after holdout) is enough for it to detect the same
# seasonality it reliably finds in the other frequencies.
DEFAULT_N = {"D": 3 * 365, "W": 5 * 52, "MS": 8 * 12, "YS": 25}

_LETTERS = list(string.ascii_uppercase)


@dataclasses.dataclass
class SeriesSpec:
    series_id: str
    category: str
    name: str
    description: str
    freq: str
    df: pd.DataFrame  # columns: ds, y, [covariate columns]
    covariate_cols: list[str]
    static_covariates: dict[str, str]
    true_price_elasticity: float | None = None
    event_points: list[int] = dataclasses.field(default_factory=list)  # index markers (e.g. shift/break points)


def _dates(n: int, freq: str, start: str = "2019-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq=freq)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _label(i: int) -> str:
    return _LETTERS[i] if i < len(_LETTERS) else f"#{i + 1}"


def _cycle_combos(count: int, *axes) -> list[tuple]:
    """Pick `count` combinations spread evenly across the full cartesian product.

    itertools.product varies its *last* axis fastest and its *first* axis
    slowest, so naively taking the first `count` combos (combos[i % len]) stays
    stuck on the first value of the first axis whenever count is smaller than
    one full "row" of that axis -- e.g. with freq as the first axis and 3 freqs
    x 3 scales x 2 trends = 18 combos, requesting 10 would silently never reach
    freq[1] or freq[2] at all. A float stride across the whole product spreads
    the sample across every axis instead, regardless of how count compares to
    the product size.
    """
    combos = list(itertools.product(*axes))
    if count >= len(combos):
        return [combos[i % len(combos)] for i in range(count)]
    step = len(combos) / count
    return [combos[int(i * step) % len(combos)] for i in range(count)]


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def make_standard(
    seed: int, freq: str = "D", n: int | None = None, scale: float = 100.0,
    trend: str = "up", seasonal: bool = True, name_suffix: str = "",
) -> SeriesSpec:
    n = n or DEFAULT_N[freq]
    rng = _rng(seed)
    t = np.arange(n)
    slope = {"up": 0.6, "down": -0.6, "flat": 0.0}[trend] * scale / n
    base = scale + slope * t
    # A year IS the base unit for annual data -- there's no sub-annual cycle to show.
    yearly = 0.15 * scale * np.sin(2 * np.pi * t / FREQ_YEARLY_PERIOD[freq]) if (seasonal and freq != "YS") else 0.0
    weekly = 0.07 * scale * np.sin(2 * np.pi * t / 7 + 1.0) if (seasonal and freq == "D") else 0.0
    noise = rng.normal(0, 0.04 * scale, n)
    y = np.maximum(base + yearly + weekly + noise, 0)
    df = pd.DataFrame({"ds": _dates(n, freq), "y": y})
    seasonal_effective = seasonal and freq != "YS"
    seas_txt = "trend + seasonality" if seasonal_effective else "trend only, no seasonality"
    return SeriesSpec(
        series_id=f"standard_{seed}",
        category="Standard",
        name=f"Standard {name_suffix} ({FREQ_LABELS[freq]})".strip(),
        description=f"{FREQ_LABELS[freq]} data, {seas_txt}, {trend} trend, scale~{scale:.0f}. The 'easy' baseline case.",
        freq=freq,
        df=df,
        covariate_cols=[],
        static_covariates={},
    )


def make_short(
    seed: int, freq: str = "D", n: int | None = None, scale: float = 50.0, trend: str = "up", name_suffix: str = "",
) -> SeriesSpec:
    default_short_n = {"D": 24, "W": 12, "MS": 8}
    n = n or default_short_n[freq]
    rng = _rng(seed)
    t = np.arange(n)
    slope = {"up": 1.5, "down": -1.5, "flat": 0.0}[trend] * scale / max(n, 1)
    noise = rng.normal(0, 0.06 * scale, n)
    y = np.maximum(scale + slope * t + noise, 0)
    df = pd.DataFrame({"ds": _dates(n, freq), "y": y})
    return SeriesSpec(
        series_id=f"short_{seed}",
        category="Short series",
        name=f"Short series {name_suffix} ({FREQ_LABELS[freq]})".strip(),
        description=f"{FREQ_LABELS[freq]} data, only {n} observations available -- tests few-shot / low-context behavior.",
        freq=freq,
        df=df,
        covariate_cols=[],
        static_covariates={},
    )


def make_level_shift(
    seed: int, freq: str = "D", n: int | None = None, scale: float = 100.0,
    trend: str = "flat", n_shifts: int | None = None, name_suffix: str = "",
) -> SeriesSpec:
    n = n or DEFAULT_N[freq]
    rng = _rng(seed)
    t = np.arange(n)
    slope = {"up": 0.3, "down": -0.3, "flat": 0.0}[trend] * scale / n
    yearly_amp = 0.1 * scale
    base = scale + slope * t + yearly_amp * np.sin(2 * np.pi * t / FREQ_YEARLY_PERIOD[freq])
    n_shifts = n_shifts if n_shifts is not None else int(rng.integers(1, 3))
    shift_points = sorted(
        rng.choice(np.arange(int(n * 0.25), int(n * 0.8)), size=n_shifts, replace=False).tolist()
    )
    level = np.zeros(n)
    for sp in shift_points:
        # Deliberately pronounced: 60-110% of scale, several multiples of the noise std.
        magnitude = rng.choice([-1, 1]) * rng.uniform(0.6, 1.1) * scale
        level[sp:] += magnitude
    noise = rng.normal(0, 0.04 * scale, n)
    y = np.maximum(base + level + noise, 0)
    df = pd.DataFrame({"ds": _dates(n, freq), "y": y})
    return SeriesSpec(
        series_id=f"levelshift_{seed}",
        category="Level shift",
        name=f"Level shift {name_suffix} ({FREQ_LABELS[freq]})".strip(),
        description=(
            f"{FREQ_LABELS[freq]} data. Abrupt structural break(s) at index {shift_points} "
            "(marked with dashed vertical lines on the chart) -- tests adaptation to sudden regime change."
        ),
        freq=freq,
        df=df,
        covariate_cols=[],
        static_covariates={},
        event_points=shift_points,
    )


def make_overlapping_cycles(
    seed: int, freq: str = "D", n: int | None = None, scale: float = 120.0, noise_level: str = "low", name_suffix: str = "",
) -> SeriesSpec:
    n = n or DEFAULT_N[freq]
    rng = _rng(seed)
    t = np.arange(n)
    if freq == "D":
        cycles = [(7, 0.08), (30.44, 0.14), (91.3, 0.10), (365.25, 0.20)]
        cycle_desc = "weekly/monthly/quarterly/yearly"
    elif freq == "W":
        cycles = [(4.345, 0.10), (13.03, 0.12), (52.18, 0.22)]
        cycle_desc = "~monthly/~quarterly/yearly"
    else:  # MS
        cycles = [(12, 0.22), (36, 0.12)]
        cycle_desc = "yearly/~3-year"
    signal = np.zeros(n)
    for i, (period, amp_frac) in enumerate(cycles):
        signal += amp_frac * scale * np.sin(2 * np.pi * t / period + i * 0.7)
    slope = 0.15 * scale / n
    trend = scale + slope * t
    noise_std = {"low": 0.04, "medium": 0.12, "high": 0.30}[noise_level] * scale
    noise = rng.normal(0, noise_std, n)
    y = np.maximum(trend + signal + noise, 0)
    df = pd.DataFrame({"ds": _dates(n, freq), "y": y})
    snr = (signal.max() - signal.min()) / max(noise_std, 1e-9)
    difficulty = {
        "low": "highly regular -- easy to extrapolate once the periods are identified",
        "medium": "moderate noise -- stresses seasonality decomposition",
        "high": "genuinely hard -- noise dominates enough that even the true periods don't fully explain the holdout",
    }[noise_level]
    return SeriesSpec(
        series_id=f"cycles_{seed}",
        category="Overlapping cycles",
        name=f"Overlapping cycles {name_suffix} ({FREQ_LABELS[freq]})".strip(),
        description=(
            f"{FREQ_LABELS[freq]} data. Superimposed {cycle_desc} cycles, {noise_level} noise "
            f"(signal:noise amplitude ratio ~{snr:.0f}:1) -- {difficulty}."
        ),
        freq=freq,
        df=df,
        covariate_cols=[],
        static_covariates={},
    )


def make_intermittent(
    seed: int, freq: str = "D", n: int | None = None, scale: float = 15.0, demand_prob: float = 0.12, name_suffix: str = "",
) -> SeriesSpec:
    n = n or DEFAULT_N[freq]
    rng = _rng(seed)
    spikes = rng.binomial(1, demand_prob, n)
    sizes = rng.gamma(shape=3.0, scale=scale / 3.0, size=n)
    y = spikes * sizes
    df = pd.DataFrame({"ds": _dates(n, freq), "y": y})
    return SeriesSpec(
        series_id=f"intermittent_{seed}",
        category="Intermittent demand",
        name=f"Intermittent demand {name_suffix} ({FREQ_LABELS[freq]})".strip(),
        description=(
            f"{FREQ_LABELS[freq]} data. Sparse, bursty demand (~{demand_prob * 100:.0f}% of periods nonzero) "
            "-- mostly zeros with occasional spikes. Stresses quantile calibration near zero."
        ),
        freq=freq,
        df=df,
        covariate_cols=[],
        static_covariates={},
    )


def make_volatility_shift(
    seed: int, freq: str = "D", n: int | None = None, scale: float = 90.0,
    direction: str = "increasing", name_suffix: str = "",
) -> SeriesSpec:
    n = n or DEFAULT_N[freq]
    rng = _rng(seed)
    t = np.arange(n)
    trend = scale + 0.08 * scale * np.sin(2 * np.pi * t / FREQ_YEARLY_PERIOD[freq])
    break_point = int(n * rng.uniform(0.35, 0.65))
    sigma_small = 0.025 * scale
    sigma_large = rng.uniform(0.30, 0.45) * scale  # deliberately pronounced: ~12-18x factor
    if direction == "increasing":
        sigma_before, sigma_after = sigma_small, sigma_large
    else:
        sigma_before, sigma_after = sigma_large, sigma_small
    sigma = np.where(t < break_point, sigma_before, sigma_after)
    noise = rng.normal(0, 1, n) * sigma
    y = np.maximum(trend + noise, 0)
    df = pd.DataFrame({"ds": _dates(n, freq), "y": y})
    factor = sigma_large / sigma_small
    verb = "increases" if direction == "increasing" else "decreases"
    return SeriesSpec(
        series_id=f"volshift_{seed}",
        category="Volatility regime change",
        name=f"Volatility regime change {name_suffix} ({FREQ_LABELS[freq]}, {direction})".strip(),
        description=(
            f"{FREQ_LABELS[freq]} data. Noise std {verb} ~{factor:.0f}x at index {break_point} "
            "(marked with a dashed vertical line) -- mean stays stable but uncertainty regime changes; "
            "good test of quantile width adaptation in both directions."
        ),
        freq=freq,
        df=df,
        covariate_cols=[],
        static_covariates={},
        event_points=[break_point],
    )


def make_exponential_growth(
    seed: int, freq: str = "D", n: int | None = None, scale: float = 20.0,
    inflection_frac: float = 0.85, name_suffix: str = "",
) -> SeriesSpec:
    """Logistic (S-curve) growth: for most of the visible history it looks exactly
    like unchecked exponential growth (the logistic function is asymptotically
    exponential well before its inflection point), then it decelerates and
    saturates near the holdout boundary -- the way real growth processes actually
    behave (adoption curves, population growth, viral spread all eventually slow
    down). Tests whether the model naively extrapolates the visible exponential
    trend into an ever-larger forecast instead of anticipating the slowdown."""
    n = n or DEFAULT_N[freq]
    rng = _rng(seed)
    t = np.arange(n)
    ceiling = scale * 14.0
    t0 = inflection_frac * n
    k = 9.0 / n
    base = ceiling / (1 + np.exp(-k * (t - t0)))
    base = np.maximum(base, scale * 0.4)  # visible floor so the early section isn't near-zero
    noise = rng.normal(0, 0.03 * scale, n)
    y = np.maximum(base + noise, 0)
    df = pd.DataFrame({"ds": _dates(n, freq), "y": y})
    inflection_idx = int(t0)
    return SeriesSpec(
        series_id=f"expgrowth_{seed}",
        category="Growth & decay traps",
        name=f"Exponential growth trap {name_suffix} ({FREQ_LABELS[freq]})".strip(),
        description=(
            f"{FREQ_LABELS[freq]} data. Looks like unchecked exponential growth through most of the "
            f"visible history, then decelerates and saturates starting around index {inflection_idx} "
            "(marked) -- no real growth process compounds forever, so this tests whether the model "
            "anticipates the slowdown instead of extrapolating the exponential trend."
        ),
        freq=freq,
        df=df,
        covariate_cols=[],
        static_covariates={},
        event_points=[inflection_idx],
    )


def make_rapid_decay(
    seed: int, freq: str = "D", n: int | None = None, scale: float = 100.0,
    decay_frac: float = 0.3, name_suffix: str = "",
) -> SeriesSpec:
    """Exponential decay toward (near) zero -- most of the drop happens within the
    first decay_frac of the series, then it stays near the floor for the rest,
    including the whole holdout. Since almost all real forecasting targets are
    non-negative, this tests whether the forecast and its quantile bands respect
    that floor instead of extrapolating the steep early decay into negative
    territory."""
    n = n or DEFAULT_N[freq]
    rng = _rng(seed)
    t = np.arange(n)
    target_frac = 0.02  # decay to ~2% of scale by decay_frac of the series
    k = -np.log(target_frac) / max(1, decay_frac * n)
    base = scale * np.exp(-k * t)
    noise_scale = 0.04 * scale * np.exp(-k * t / 2)  # noise shrinks with the level, doesn't swamp the tail
    noise = rng.normal(0, 1, n) * noise_scale
    y = np.maximum(base + noise, 0)
    df = pd.DataFrame({"ds": _dates(n, freq), "y": y})
    decay_idx = int(decay_frac * n)
    return SeriesSpec(
        series_id=f"rapiddecay_{seed}",
        category="Growth & decay traps",
        name=f"Rapid decay {name_suffix} ({FREQ_LABELS[freq]})".strip(),
        description=(
            f"{FREQ_LABELS[freq]} data. Exponential decay from {scale:.0f} down to near-zero by index "
            f"{decay_idx} (marked), then flat near the floor -- tests whether the forecast and its "
            "quantile bands respect the non-negativity floor instead of extrapolating the steep early "
            "decay into negative values."
        ),
        freq=freq,
        df=df,
        covariate_cols=[],
        static_covariates={},
        event_points=[decay_idx],
    )


def make_retail_covariates(
    seed: int, freq: str = "D", n: int | None = None, base_price: float = 10.0, base_sales: float = 200.0,
    elasticity_beta: float = -18.0, store_type: str = "standard", name_suffix: str = "",
) -> SeriesSpec:
    n = n or DEFAULT_N[freq]
    rng = _rng(seed)
    t = np.arange(n)
    trend = base_sales + (0.3 * base_sales / n) * t
    yearly = 0.12 * base_sales * np.sin(2 * np.pi * t / FREQ_YEARLY_PERIOD[freq])
    weekly = 0.06 * base_sales * np.sin(2 * np.pi * t / 7 + 0.8) if freq == "D" else 0.0

    price_walk = np.cumsum(rng.normal(0, 0.05, n))
    price = np.clip(base_price + price_walk * 0.3, base_price * 0.7, base_price * 1.3)
    price_effect = elasticity_beta * (price - base_price)

    promo = (rng.random(n) < 0.10).astype(float)
    promo_effect = 0.3 * base_sales * promo

    holiday_gap = {"D": 60, "W": 10, "MS": 3}[freq]
    holiday = np.zeros(n)
    holiday_days = rng.choice(n, size=max(2, n // holiday_gap), replace=False)
    holiday[holiday_days] = 1.0
    holiday_effect = 0.45 * base_sales * holiday

    noise = rng.normal(0, 0.03 * base_sales, n)
    y = np.maximum(trend + yearly + weekly + price_effect + promo_effect + holiday_effect + noise, 0)

    df = pd.DataFrame(
        {"ds": _dates(n, freq), "y": y, "price": price, "promo": promo, "holiday": holiday}
    )
    mean_early_sales = float(np.mean(y[: max(10, n // 6)]))
    true_elasticity = elasticity_beta * (base_price / mean_early_sales)

    return SeriesSpec(
        series_id=f"retail_{seed}",
        category="Covariates / elasticity",
        name=f"Retail store {name_suffix} ({store_type.title()}, {FREQ_LABELS[freq]})".strip(),
        description=(
            f"{FREQ_LABELS[freq]} data. {store_type.title()} store, base price ${base_price:.2f}, "
            f"true price elasticity {true_elasticity:.2f}. "
            "Covariates: price ($, numeric), promo (0/1 flag), holiday (0/1 flag)."
        ),
        freq=freq,
        df=df,
        covariate_cols=["price", "promo", "holiday"],
        static_covariates={"store_type": store_type},
        true_price_elasticity=true_elasticity,
    )


# ---------------------------------------------------------------------------
# Catalog: ~10 instances per category, spanning frequency/scale/trend/etc.
# ---------------------------------------------------------------------------


def _build_catalog() -> dict[str, SeriesSpec]:
    catalog: dict[str, SeriesSpec] = {}
    N = 10

    # Standard: spread round-robin across daily/weekly/monthly/annual (stays balanced
    # regardless of N), varying scale/trend/seasonality across instances.
    std_freqs = ("D", "W", "MS", "YS")
    for i, (scale, trend, seasonal) in enumerate(
        _cycle_combos(N, (20.0, 100.0, 600.0), ("up", "down", "flat"), (True, False))
    ):
        freq = std_freqs[i % len(std_freqs)]
        s = make_standard(seed=1000 + i, freq=freq, scale=scale, trend=trend, seasonal=seasonal, name_suffix=_label(i))
        catalog[s.series_id] = s

    for i, (freq, scale, trend) in enumerate(
        _cycle_combos(N, ("D", "W", "MS"), (10.0, 50.0, 300.0), ("up", "down", "flat"))
    ):
        s = make_short(seed=2000 + i, freq=freq, scale=scale, trend=trend, name_suffix=_label(i))
        catalog[s.series_id] = s

    for i, (freq, scale, trend, n_shifts) in enumerate(
        _cycle_combos(N, ("D", "W", "MS"), (30.0, 100.0, 400.0), ("flat", "up"), (1, 2))
    ):
        s = make_level_shift(seed=3000 + i, freq=freq, scale=scale, trend=trend, n_shifts=n_shifts, name_suffix=_label(i))
        catalog[s.series_id] = s

    for i, (freq, scale, noise_level) in enumerate(
        _cycle_combos(N, ("D", "W", "MS"), (50.0, 150.0, 500.0), ("low", "medium", "high"))
    ):
        s = make_overlapping_cycles(seed=4000 + i, freq=freq, scale=scale, noise_level=noise_level, name_suffix=_label(i))
        catalog[s.series_id] = s

    for i, (freq, scale, demand_prob) in enumerate(
        _cycle_combos(N, ("D", "W", "MS"), (10.0, 30.0, 100.0), (0.08, 0.15, 0.25))
    ):
        s = make_intermittent(seed=5000 + i, freq=freq, scale=scale, demand_prob=demand_prob, name_suffix=_label(i))
        catalog[s.series_id] = s

    for i, (freq, scale) in enumerate(
        _cycle_combos(N, ("D", "W", "MS"), (30.0, 90.0, 250.0, 600.0, 1000.0))
    ):
        # Alternate direction so exactly half the gallery shows volatility falling, not
        # just rising -- a model that only ever widens its bands never gets tested on
        # correctly narrowing them again.
        direction = "increasing" if i % 2 == 0 else "decreasing"
        s = make_volatility_shift(seed=6000 + i, freq=freq, scale=scale, direction=direction, name_suffix=_label(i))
        catalog[s.series_id] = s

    for i, (freq, scale, inflection_frac) in enumerate(
        _cycle_combos(N // 2, ("D", "W", "MS"), (10.0, 30.0, 80.0), (0.80, 0.85, 0.90))
    ):
        s = make_exponential_growth(seed=8000 + i, freq=freq, scale=scale, inflection_frac=inflection_frac, name_suffix=_label(i))
        catalog[s.series_id] = s

    # decay_frac positioned close to the gallery's default 25% holdout (i.e. holdout
    # starts around 75% through the series) -- otherwise the decay finishes long
    # before the holdout window even begins, and forecasting "stays flat at zero"
    # is trivial. Placing the event just before/at the holdout boundary means the
    # forecast has to pick up right where the decay is still happening or just
    # finished, which is a much harder and more relevant test of the floor.
    for i, (freq, scale, decay_frac) in enumerate(
        _cycle_combos(N // 2, ("D", "W", "MS"), (50.0, 150.0, 400.0), (0.55, 0.65, 0.75))
    ):
        s = make_rapid_decay(seed=8100 + i, freq=freq, scale=scale, decay_frac=decay_frac, name_suffix=_label(i))
        catalog[s.series_id] = s

    store_types = ["premium", "standard", "discount", "boutique", "warehouse"]
    for i, (freq, base_price, base_sales, elasticity_beta, store_type) in enumerate(
        _cycle_combos(N, ("D", "W"), (5.0, 9.0, 14.0, 25.0), (150.0, 260.0, 500.0), (-8.0, -18.0, -35.0), store_types)
    ):
        s = make_retail_covariates(
            seed=7000 + i, freq=freq, base_price=base_price, base_sales=base_sales,
            elasticity_beta=elasticity_beta, store_type=store_type, name_suffix=_label(i),
        )
        catalog[s.series_id] = s

    return catalog


@lru_cache(maxsize=1)
def get_catalog() -> dict[str, SeriesSpec]:
    return _build_catalog()


def list_categories() -> list[str]:
    seen: list[str] = []
    for spec in get_catalog().values():
        if spec.category not in seen:
            seen.append(spec.category)
    return seen


def get_series(series_id: str) -> SeriesSpec:
    return get_catalog()[series_id]


def series_ids_in_category(category: str) -> list[str]:
    return [sid for sid, spec in get_catalog().items() if spec.category == category]
