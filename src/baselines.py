"""Classical statistical forecasting baselines via Nixtla's statsforecast, for
comparison against TimesFM on the same series/metrics.

statsforecast expresses uncertainty as symmetric prediction-interval `level`s
rather than raw quantiles. Requesting levels [20, 40, 60, 80] yields exactly
the deciles (10/20/30/40/60/70/80/90) that TimesFM's quantile head returns,
plus the point forecast as the median (50th), so the two can be compared
quantile-for-quantile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, MSTL, SeasonalNaive, Theta

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
MODEL_CHOICES = ["SeasonalNaive", "AutoARIMA", "AutoETS", "Theta", "MSTL", *REGRESSION_MODEL_CHOICES]

# What each baseline actually does and doesn't account for -- e.g. SeasonalNaive
# has no idea trend exists, which is exactly the kind of detail that isn't
# obvious from the name alone. Surfaced as a help tooltip next to each model's
# checkbox in the UI.
MODEL_INFO: dict[str, str] = {
    "SeasonalNaive": (
        "Repeats the value from exactly one season ago (e.g. same weekday last week) with no change. "
        "Does NOT account for trend, and has no smoothing or noise-averaging at all -- if last season's "
        "value was an outlier, that outlier is the forecast. A simple, often surprisingly tough-to-beat "
        "benchmark precisely because it makes no assumptions that can be wrong."
    ),
    "AutoARIMA": (
        "Automatically searches (S)ARIMA orders (differencing + autoregressive + moving-average terms, "
        "seasonal and non-seasonal) via AIC and fits the best one. Differencing lets it capture a trend "
        "implicitly and adapt if the trend changes; autoregressive terms capture short-term "
        "autocorrelation. Relatively slow to fit because of the order search."
    ),
    "AutoETS": (
        "Automatically searches Error-Trend-Season exponential smoothing specifications (additive/"
        "multiplicative/none for each component, trend damped or not) via AIC and fits the best one. "
        "Explicitly models level, trend, and seasonality as smoothly-updating states -- but needs enough "
        "full seasonal cycles of history to justify fitting a seasonal term at all; with too little "
        "history it will correctly fall back to a non-seasonal fit rather than guess."
    ),
    "Theta": (
        "Decomposes the series into a long-term trend line and a short-term component, forecasts each "
        "separately (the trend linearly, the short-term component via simple exponential smoothing), "
        "then recombines them. A deceptively simple method that won the M3 forecasting competition; "
        "captures trend but has a fairly rigid, linear view of it."
    ),
    "MSTL": (
        "Decomposes the series into trend + multiple seasonal components (e.g. weekly AND yearly at "
        "once, unlike the other models here which only ever fit one seasonal period) via repeated STL "
        "smoothing, forecasts the trend+remainder, and adds the seasonal components back. Needs at least "
        "~2 full cycles of each seasonal period to be fit at all -- periods without enough history are "
        "silently dropped rather than guessed at."
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
        "is extrapolated as a straight line, so it will be systematically wrong for curved growth/decay."
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

# Default seasonal period (in native units of the series' own frequency) per freq.
# YS (annual) has no sub-period cycle -- a year IS the base unit -- so season_length=1
# (non-seasonal fitting), same convention these libraries use for yearly data.
DEFAULT_SEASON_LENGTH = {"D": 7, "W": 52, "MS": 12, "YS": 1}


def _season_lengths(n: int, freq: str) -> list[int]:
    """Candidate MSTL seasonal periods for this frequency, filtered to ones
    the series actually has enough data to estimate (>= ~2 full cycles)."""
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
        return []  # no sub-annual cycle -- MSTL isn't meaningful here
    return [DEFAULT_SEASON_LENGTH.get(freq, 7)]


def _build_model(name: str, n: int, freq: str):
    sl = DEFAULT_SEASON_LENGTH.get(freq, 7)
    if name == "SeasonalNaive":
        return SeasonalNaive(season_length=sl)
    if name == "AutoARIMA":
        return AutoARIMA(season_length=sl)
    if name == "AutoETS":
        return AutoETS(season_length=sl)
    if name == "Theta":
        return Theta(season_length=sl)
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
    season_length = DEFAULT_SEASON_LENGTH.get(freq, 7)

    for name in model_names:
        if name in REGRESSION_MODEL_CHOICES:
            try:
                if name == "ExponentialTrend":
                    results[name] = _fit_exponential_trend_baseline(df["y"].to_numpy(), horizon)
                else:
                    results[name] = _fit_regression_baseline(
                        df["y"].to_numpy(), horizon, season_length, with_seasonal=(name == "LinearRegression")
                    )
            except Exception as exc:  # noqa: BLE001 - surfaced in UI, not swallowed silently
                results[name] = {"point": None, "quantiles": None, "error": str(exc)}
            continue
        try:
            model = _build_model(name, n, freq)
            sf = StatsForecast(models=[model], freq=freq, n_jobs=1)
            fc = sf.forecast(df=sf_df, h=horizon, level=LEVELS)
            point = fc[name].to_numpy()
            quantiles: dict[float, np.ndarray] = {0.5: point.copy()}
            for level, (qlo, qhi) in _LEVEL_TO_QUANTILES.items():
                quantiles[qlo] = fc[f"{name}-lo-{level}"].to_numpy()
                quantiles[qhi] = fc[f"{name}-hi-{level}"].to_numpy()
            results[name] = {"point": point, "quantiles": quantiles, "error": None}
        except Exception as exc:  # noqa: BLE001 - surfaced in UI, not swallowed silently
            results[name] = {"point": None, "quantiles": None, "error": str(exc)}

    return results
