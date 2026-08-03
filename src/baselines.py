"""Classical statistical forecasting baselines via Nixtla's statsforecast, for
comparison against TimesFM on the same series/metrics.

statsforecast expresses uncertainty as symmetric prediction-interval `level`s
rather than raw quantiles. Requesting levels [20, 40, 60, 80] yields exactly
the deciles (10/20/30/40/60/70/80/90) that TimesFM's quantile head returns,
plus the point forecast as the median (50th), so the two can be compared
quantile-for-quantile.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, MSTL, SeasonalNaive, Theta
from statsmodels.tsa.seasonal import STL, seasonal_decompose

LEVELS = [20, 40, 60, 80]

# level -> (lower quantile, upper quantile), derived from symmetric PI construction
_LEVEL_TO_QUANTILES = {20: (0.4, 0.6), 40: (0.3, 0.7), 60: (0.2, 0.8), 80: (0.1, 0.9)}

DECILES = (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9)

# Classical regression baselines (fit directly with statsmodels OLS, not via
# statsforecast/StatsForecast) alongside the state-space/differencing models
# above -- a different family entirely: extrapolate a fitted deterministic
# function of time rather than model the series' own autocorrelation structure.
# LinearTrend/LinearRegression fit on the raw level, so they extrapolate a
# straight line -- systematically wrong for a curved exponential growth/decay
# shape. ExponentialTrend instead fits on log(y), which turns exponential
# growth/decay into a straight line, then exponentiates the forecast back --
# the standard classical fix for exactly that shape (see the "Growth & decay
# traps" gallery category, which is built to expose this difference).
REGRESSION_MODEL_CHOICES = ["LinearTrend", "LinearRegression", "ExponentialTrend"]
MODEL_CHOICES = ["SeasonalNaive", "AutoETS", "Theta", "MSTL", *REGRESSION_MODEL_CHOICES]

# What each baseline actually does and doesn't account for -- e.g. SeasonalNaive
# has no idea trend exists, which is exactly the kind of detail that isn't
# obvious from the name alone. Surfaced as a help tooltip next to each model's
# checkbox in the UI.
MODEL_INFO: dict[str, str] = {
    "SeasonalNaive": (
        "Repeats the value from exactly one season ago with no change. Does NOT account for trend, and "
        "has no smoothing or noise-averaging at all -- if last season's value was an outlier, that "
        "outlier is the forecast. A simple, often surprisingly tough-to-beat benchmark precisely because "
        "it makes no assumptions that can be wrong. The season length itself is detected per series (see "
        "detect_season_length) rather than assumed from frequency alone -- see the lag shown next to its "
        "name above."
    ),
    "AutoETS": (
        "Automatically searches Error-Trend-Season exponential smoothing specifications (additive/"
        "multiplicative/none for each component, trend damped or not) via AIC and fits the best one. "
        "Explicitly models level, trend, and seasonality as smoothly-updating states -- but needs enough "
        "full seasonal cycles of history to justify fitting a seasonal term at all; with too little "
        "history it will correctly fall back to a non-seasonal fit rather than guess. Uses the same "
        "detected season length as SeasonalNaive/Theta (see detect_season_length); benchmarked directly, "
        "fitting at a long (e.g. 365-day) detected period is ~40x slower than a short one (18.5s vs "
        "0.45s per series) -- noticeably slower on strongly seasonal daily data, but still tractable, "
        "unlike AutoARIMA at the same period (dropped from this app for exactly that reason -- its order "
        "search didn't finish a single fit in over 6 minutes at a 365-day period)."
    ),
    "Theta": (
        "Decomposes the series into a long-term trend line and a short-term component, forecasts each "
        "separately (the trend linearly, the short-term component via simple exponential smoothing), "
        "then recombines them. A deceptively simple method that won the M3 forecasting competition; "
        "captures trend but has a fairly rigid, linear view of it. Uses the same detected season length "
        "as SeasonalNaive/AutoETS (see detect_season_length) -- fitting cost doesn't depend on the "
        "period, so there's no tradeoff to make here."
    ),
    "MSTL": (
        "Decomposes the series into trend + multiple seasonal components (e.g. weekly AND yearly at "
        "once, unlike the other models here which only ever fit one seasonal period) via repeated STL "
        "smoothing, forecasts the trend+remainder, and adds the seasonal components back. Needs at least "
        "~2 full cycles of each seasonal period to be fit at all -- periods without enough history are "
        "silently dropped rather than guessed at. The only baseline here that isn't forced to pick just "
        "one period at daily frequency, so it tends to have an edge on the 'Overlapping cycles' category "
        "specifically."
    ),
    "LinearTrend": (
        "OLS regression of the series on a straight time trend only -- no seasonality, no "
        "autocorrelation modeling of any kind. Extrapolates a straight line forever, so it's "
        "systematically wrong for any curved trend (e.g. exponential growth/decay, or a trend that's "
        "decelerating) -- see ExponentialTrend for that case."
    ),
    "LinearRegression": (
        "OLS regression of the series on a straight time trend plus two Fourier seasonal harmonics -- "
        "captures a single repeating cycle shape on top of a trend. Like LinearTrend, the trend itself "
        "is extrapolated as a straight line, so it will be systematically wrong for curved growth/decay. "
        "Uses the same detected season length as SeasonalNaive/AutoETS/Theta (see detect_season_length) "
        "for its Fourier period."
    ),
    "ExponentialTrend": (
        "OLS regression of log(series) on a straight time trend, forecast by exponentiating back. A "
        "straight line in log-space IS exponential growth/decay in the original units, so this is the "
        "classical fix for a series that grows or shrinks multiplicatively (see the 'Growth & decay "
        "traps' gallery). No seasonal term and no autocorrelation modeling; its quantile band is "
        "intentionally right-skewed (wider above the point forecast than below), unlike the other "
        "regression baselines' symmetric bands."
    ),
}

# Seasonal period (in native units of the series' own frequency) used ONLY for the
# MASE/RMSSE scaling denominator (naive_seasonal_scale in metrics.py) -- a fixed,
# per-frequency convention, independent of whichever period a given model actually
# fits with. YS (annual) has no sub-period cycle -- a year IS the base unit -- so
# season_length=1 (non-seasonal scaling), same convention these libraries use for
# yearly data.
DEFAULT_SEASON_LENGTH = {"D": 7, "W": 52, "MS": 12, "YS": 1}


def _season_lengths(n: int, freq: str) -> list[int]:
    """Candidate seasonal periods for this frequency, filtered to ones the
    series actually has enough data to estimate (>= ~2 full cycles). Used both
    as MSTL's multi-period list and as the candidate set detect_season_length
    picks a single winner from."""
    if freq == "D":
        lengths = [7]
        if n >= 90:
            lengths.append(30)
        if n >= 750:
            lengths.append(365)
        return lengths
    if freq == "W":
        lengths = []
        if n >= 12:
            lengths.append(4)
        if n >= 120:
            lengths.append(52)
        return lengths
    if freq == "MS":
        lengths = []
        if n >= 24:
            lengths.append(12)
        return lengths
    if freq == "YS":
        return []  # no sub-annual cycle -- seasonality isn't meaningful here
    return [DEFAULT_SEASON_LENGTH.get(freq, 7)]


# Below this, a candidate period is treated as "not really seasonal" rather than
# just the least-bad option. There's no single universally agreed cutoff for this
# statistic (Wang/Smith/Hyndman's seasonal-strength feature is usually used as a
# continuous score, not thresholded), but a series that clears ~0.3 has a clearly
# visible, repeatable cycle at that period, while noise typically scores well under
# 0.1 -- picked to separate "real cycle" from "STL found a pattern in the noise."
SEASONAL_STRENGTH_THRESHOLD = 0.3


def _seasonal_strength(y: np.ndarray, period: int) -> float:
    """Hyndman & Athanasopoulos's seasonal-strength measure: how much of the
    variance left after removing trend is explained by the seasonal component,
    via an STL decomposition at this candidate period -- 0 means no seasonality,
    approaching 1 means strongly seasonal. This is the standard, well-established
    way to test whether a candidate period is real, rather than assuming it from
    frequency alone (see detect_season_length)."""
    if period < 2 or len(y) < 2 * period:
        return 0.0
    try:
        result = STL(y, period=period, robust=True).fit()
    except Exception:
        return 0.0
    seasonal_and_resid_var = np.var(result.seasonal + result.resid)
    if seasonal_and_resid_var < 1e-12:
        return 0.0
    return float(max(0.0, 1.0 - np.var(result.resid) / seasonal_and_resid_var))


def detect_season_length(y: np.ndarray, freq: str) -> int:
    """Picks ONE season_length for this series, shared by every single-season
    model (SeasonalNaive, AutoETS, Theta, LinearRegression's Fourier term) --
    via seasonal-strength testing across this frequency's candidate periods
    (_season_lengths), rather than a fixed per-model guess or a per-model cost
    tradeoff. Falls back to 1 (non-seasonal) if no candidate clears
    SEASONAL_STRENGTH_THRESHOLD, instead of defaulting to the shortest
    candidate just because it's conventional. MSTL is the one baseline that
    doesn't use this -- it fits every candidate period at once instead of
    picking a single winner.

    Mostly-zero series (intermittent demand) skip detection entirely and go
    straight to 1: verified directly that STL reports spuriously HIGH seasonal
    strength (~0.96 at a 365-day period) on this app's Intermittent demand
    category, which has no designed periodicity at all -- with so few nonzero
    points, STL's seasonal component ends up fitting the sparse spike pattern
    itself rather than a real repeating cycle, especially with only ~3 annual
    cycles of history to check it against.
    """
    if np.mean(y == 0) > 0.5:
        return 1
    candidates = _season_lengths(len(y), freq)
    if not candidates:
        return 1
    scored = [(p, _seasonal_strength(y, p)) for p in candidates]
    best_period, best_strength = max(scored, key=lambda ps: ps[1])
    return best_period if best_strength >= SEASONAL_STRENGTH_THRESHOLD else 1


def _build_model(name: str, n: int, freq: str, season_length: int):
    if name == "SeasonalNaive":
        return SeasonalNaive(season_length=season_length)
    if name == "AutoETS":
        # AutoETS's default model="ZZZ" (full auto search over error/trend/season)
        # is verified to sometimes discard a real, strong seasonal signal: on this
        # app's own standard_1000 series, 'ZZZ' picked ETS(A,Ad,N) at AIC=1167.5,
        # while forcing an additive season with model="ZZA" found ETS(A,A,A) at
        # AIC=1155.9 -- decisively better, just never reached by the full search.
        # Since we've already independently confirmed real seasonality via
        # detect_season_length's STL-based test whenever season_length > 1, force
        # the search to use it rather than risk 'ZZZ' silently dropping it again.
        model_spec = "ZZA" if season_length > 1 else "ZZZ"
        return AutoETS(season_length=season_length, model=model_spec)
    if name == "Theta":
        return Theta(season_length=season_length)
    if name == "MSTL":
        lengths = _season_lengths(n, freq)
        if not lengths:
            reason = "no sub-period cycle at annual frequency" if freq == "YS" else f"not enough history (n={n})"
            raise ValueError(f"MSTL not applicable: {reason}")
        return MSTL(season_length=lengths)
    raise ValueError(f"Unknown model: {name}")


def _regression_design(t: np.ndarray, season_length: int, with_seasonal: bool) -> pd.DataFrame:
    X = pd.DataFrame({"t": t})
    if with_seasonal and season_length > 1:
        for k in (1, 2):  # two harmonics -- enough to capture a typical single-cycle shape
            X[f"sin{k}"] = np.sin(2 * np.pi * k * t / season_length)
            X[f"cos{k}"] = np.cos(2 * np.pi * k * t / season_length)
    return sm.add_constant(X, has_constant="add")


def _fit_regression_baseline(y: np.ndarray, horizon: int, season_length: int, with_seasonal: bool) -> dict:
    """OLS regression of y on a linear time trend (+ two Fourier seasonal
    harmonics if with_seasonal), forecast by extrapolating the fitted line/curve.
    Quantiles come from a normal approximation using the regression's own
    observation-level prediction standard error (statsmodels' `se_obs`), which
    widens appropriately with the forecast horizon and residual noise -- unlike
    ETS/ARIMA, this treats the series as a deterministic function of time plus
    noise, with no autocorrelation modeling at all.
    """
    n = len(y)
    if with_seasonal and n <= 2 * max(season_length, 1):
        raise ValueError(f"Not enough history for a seasonal regression (n={n}, season_length={season_length})")
    X_train = _regression_design(np.arange(n), season_length, with_seasonal)
    model = sm.OLS(y, X_train).fit()
    X_future = _regression_design(np.arange(n, n + horizon), season_length, with_seasonal)
    pred = model.get_prediction(X_future)
    point = np.asarray(pred.predicted_mean)
    se_obs = np.asarray(pred.se_obs)
    quantiles: dict[float, np.ndarray] = {0.5: point.copy()}
    for q in DECILES:
        quantiles[q] = point + stats.norm.ppf(q) * se_obs
    return {"point": point, "quantiles": quantiles, "error": None}


def _fit_exponential_trend_baseline(y: np.ndarray, horizon: int) -> dict:
    """OLS regression of log(y) on a linear time trend, forecast by
    exponentiating back. Unlike LinearTrend (which fits the raw level and
    extrapolates a straight line -- systematically wrong for a curved
    exponential shape), a straight line in log-space *is* exponential growth
    or decay in the original units, so this is the standard classical fix for
    a series that grows/decays multiplicatively. Quantile bounds are
    exponentiated too, which makes them right-skewed on the original scale
    (wider above the point forecast than below) -- appropriate for a
    multiplicative process, unlike the symmetric bands the level-space
    regressions produce. No seasonal option: none of this app's exponential
    growth/decay series have a seasonal component to speak of.
    """
    y_floor = np.maximum(y, 1e-6)  # log(0) is undefined; the rapid-decay series legitimately hit 0
    log_y = np.log(y_floor)
    n = len(y)
    X_train = _regression_design(np.arange(n), season_length=1, with_seasonal=False)
    model = sm.OLS(log_y, X_train).fit()
    X_future = _regression_design(np.arange(n, n + horizon), season_length=1, with_seasonal=False)
    pred = model.get_prediction(X_future)
    log_point = np.asarray(pred.predicted_mean)
    log_se = np.asarray(pred.se_obs)
    quantiles: dict[float, np.ndarray] = {0.5: np.exp(log_point)}
    for q in DECILES:
        quantiles[q] = np.exp(log_point + stats.norm.ppf(q) * log_se)
    return {"point": np.exp(log_point), "quantiles": quantiles, "error": None}


def _fit_theta_forced_seasonal(df: pd.DataFrame, horizon: int, freq: str, season_length: int) -> dict:
    """Wraps nixtla's Theta with our own pre-deseasonalization, bypassing its
    built-in seasonality test -- verified directly to be unreliable on trending
    data: it only checks whether the ACF *at exactly lag season_length* clears a
    Bartlett-formula threshold, which on this app's own standard_1000 series
    scored a ratio of 1.25 against a 1.645 cutoff (not "significant"), even
    though detect_season_length's STL-based test found a strength of 0.898 --
    unambiguously seasonal. A trend inflates ACF at every lag, which is exactly
    what makes this kind of undetrended, single-lag test unreliable.

    Since we've already independently confirmed real seasonality whenever this
    function is called (season_length > 1), decompose it ourselves with the
    same statsmodels routine Theta would have used internally, fit plain Theta
    on the deseasonalized series, then add the (repeating) seasonal component
    back onto both the point forecast and every quantile -- shifting the whole
    band together, not just the point.
    """
    y = df["y"].to_numpy()
    seasonal = np.asarray(
        seasonal_decompose(y, model="additive", period=season_length, extrapolate_trend="freq").seasonal
    )
    sf_df = pd.DataFrame({"unique_id": "series", "ds": df["ds"].values, "y": y - seasonal})
    sf = StatsForecast(models=[Theta(season_length=1)], freq=freq, n_jobs=1)
    fc = sf.forecast(df=sf_df, h=horizon, level=LEVELS)
    # The last full cycle of `seasonal` is already phase-aligned to end at the
    # series' last observed point, so tiling it forward reproduces the correct
    # phase for every future step.
    future_seasonal = seasonal[-season_length:][np.arange(horizon) % season_length]
    point = fc["Theta"].to_numpy() + future_seasonal
    quantiles: dict[float, np.ndarray] = {0.5: point.copy()}
    for level, (qlo, qhi) in _LEVEL_TO_QUANTILES.items():
        quantiles[qlo] = fc[f"Theta-lo-{level}"].to_numpy() + future_seasonal
        quantiles[qhi] = fc[f"Theta-hi-{level}"].to_numpy() + future_seasonal
    return {"point": point, "quantiles": quantiles, "error": None}


def run_baselines(
    df: pd.DataFrame, horizon: int, model_names: list[str] | None = None, freq: str = "D"
) -> dict[str, dict]:
    """Fit + forecast each requested classical model on df (columns: ds, y).

    Returns {model_name: {"point": np.ndarray, "quantiles": {q: np.ndarray}, "error": str|None}}
    Models that fail (e.g. not enough data) return an "error" message instead of raising.
    """
    model_names = model_names or MODEL_CHOICES
    n = len(df)
    results: dict[str, dict] = {}

    sf_df = pd.DataFrame({"unique_id": "series", "ds": df["ds"].values, "y": df["y"].values})
    # One season_length per series, detected once (see detect_season_length) and
    # shared by every single-season model below -- SeasonalNaive, AutoETS, Theta,
    # and LinearRegression's Fourier term -- instead of each picking independently.
    season_length = detect_season_length(df["y"].to_numpy(), freq)

    def _tag_season_length(name: str) -> int | list[int] | None:
        if name == "MSTL":
            return _season_lengths(n, freq) or None
        if name in ("SeasonalNaive", "AutoETS", "Theta", "LinearRegression"):
            return season_length
        return None  # LinearTrend, ExponentialTrend -- no seasonal term

    for name in model_names:
        if name in REGRESSION_MODEL_CHOICES:
            t0 = time.perf_counter()
            try:
                if name == "ExponentialTrend":
                    results[name] = _fit_exponential_trend_baseline(df["y"].to_numpy(), horizon)
                else:
                    results[name] = _fit_regression_baseline(
                        df["y"].to_numpy(), horizon, season_length, with_seasonal=(name == "LinearRegression")
                    )
                results[name]["calc_ms"] = (time.perf_counter() - t0) * 1000
            except Exception as exc:  # noqa: BLE001 - surfaced in UI, not swallowed silently
                results[name] = {"point": None, "quantiles": None, "error": str(exc)}
            results[name]["season_length"] = _tag_season_length(name)
            continue
        t0 = time.perf_counter()
        try:
            if name == "Theta" and season_length > 1:
                # Bypasses Theta's own unreliable seasonality test -- see
                # _fit_theta_forced_seasonal's docstring for the verified failure.
                results[name] = _fit_theta_forced_seasonal(df, horizon, freq, season_length)
            else:
                model = _build_model(name, n, freq, season_length)
                sf = StatsForecast(models=[model], freq=freq, n_jobs=1)
                fc = sf.forecast(df=sf_df, h=horizon, level=LEVELS)
                point = fc[name].to_numpy()
                quantiles: dict[float, np.ndarray] = {0.5: point.copy()}
                for level, (qlo, qhi) in _LEVEL_TO_QUANTILES.items():
                    quantiles[qlo] = fc[f"{name}-lo-{level}"].to_numpy()
                    quantiles[qhi] = fc[f"{name}-hi-{level}"].to_numpy()
                results[name] = {"point": point, "quantiles": quantiles, "error": None}
            results[name]["calc_ms"] = (time.perf_counter() - t0) * 1000
        except Exception as exc:  # noqa: BLE001 - surfaced in UI, not swallowed silently
            results[name] = {"point": None, "quantiles": None, "error": str(exc)}
        results[name]["season_length"] = _tag_season_length(name)

    return results
