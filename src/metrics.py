"""Forecast accuracy metrics: point error, pinball (quantile) loss, and interval coverage."""

from __future__ import annotations

import numpy as np


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def mape(actual: np.ndarray, forecast: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(actual), eps)
    return float(np.mean(np.abs(actual - forecast) / denom) * 100)


def smape(actual: np.ndarray, forecast: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum((np.abs(actual) + np.abs(forecast)) / 2, eps)
    return float(np.mean(np.abs(actual - forecast) / denom) * 100)


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def pinball_loss(actual: np.ndarray, quantile_forecast: np.ndarray, q: float) -> float:
    """Quantile (pinball) loss for a single quantile level q in (0, 1)."""
    diff = actual - quantile_forecast
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def mean_pinball_loss(actual: np.ndarray, quantiles: dict[float, np.ndarray]) -> float:
    """Average pinball loss across a dict of {quantile_level: forecast_array}."""
    losses = [pinball_loss(actual, fc, q) for q, fc in quantiles.items()]
    return float(np.mean(losses))


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of actuals falling within [lower, upper]."""
    inside = (actual >= lower) & (actual <= upper)
    return float(np.mean(inside)) * 100


def coverage_se(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Standard error of the coverage % above, treating each holdout point's
    in/out-of-interval status as a Bernoulli trial (the standard binomial-
    proportion SE, sqrt(p(1-p)/n)). This is an approximation -- consecutive
    points in a time series aren't fully independent -- but it gives a useful
    sense of how much a coverage estimate from a short holdout should be
    trusted: e.g. "78% coverage" from just 10 points has a huge SE and could
    easily really be anywhere from 55-95%, while "78%" from 200 points is a
    much tighter, more trustworthy estimate."""
    inside = (actual >= lower) & (actual <= upper)
    p = float(np.mean(inside))
    n = len(actual)
    return float(np.sqrt(p * (1 - p) / n) * 100) if n > 0 else float("nan")


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))


# ---------------------------------------------------------------------------
# Scaled errors (Hyndman & Koehler 2006; headline metrics of the M4 and M5
# forecasting competitions). MAE/RMSE/MAPE are all either scale-dependent (MAE,
# RMSE -- not comparable across series of different magnitude, so averaging them
# across a mixed set of series is close to meaningless) or ill-behaved near zero
# (MAPE). MASE and RMSSE fix both problems by scaling the error against the
# in-sample error of a simple seasonal-naive one-step-ahead forecast: they are
# scale-free (so averaging across series/categories of wildly different scale is
# meaningful) and well-defined even when actuals are zero (the M5 series -- retail
# unit sales -- are full of zeros, which is exactly why M5 adopted this family).
# A value < 1 means "beats seasonal-naive on this series"; > 1 means "loses to it".
# ---------------------------------------------------------------------------


def naive_seasonal_scale(train_y: np.ndarray, season_length: int = 1) -> float:
    """In-sample MAE of a one-step-ahead seasonal-naive forecast (y_t vs y_{t-m}) --
    the MASE scaling denominator. Falls back to m=1 (first differences) if there
    isn't enough history for the requested seasonal lag, and to a constant if the
    series is degenerate (too short, or perfectly flat)."""
    m = season_length if len(train_y) > season_length else 1
    if len(train_y) <= m:
        return 1.0
    scale = float(np.mean(np.abs(train_y[m:] - train_y[:-m])))
    return scale if scale > 1e-8 else 1.0


def naive_seasonal_scale_sq(train_y: np.ndarray, season_length: int = 1) -> float:
    """In-sample MSE of a one-step-ahead seasonal-naive forecast -- the RMSSE
    scaling denominator (same idea as naive_seasonal_scale, squared errors)."""
    m = season_length if len(train_y) > season_length else 1
    if len(train_y) <= m:
        return 1.0
    scale = float(np.mean((train_y[m:] - train_y[:-m]) ** 2))
    return scale if scale > 1e-8 else 1.0


def mase(actual: np.ndarray, forecast: np.ndarray, train_y: np.ndarray, season_length: int = 1) -> float:
    """Mean Absolute Scaled Error -- the M4 competition's headline point-forecast
    metric. Scale-free and defined even at actual=0, so (unlike MAE/RMSE/MAPE) it's
    safe to average across series of different scale, frequency, or sparsity."""
    return mae(actual, forecast) / naive_seasonal_scale(train_y, season_length)


def rmsse(actual: np.ndarray, forecast: np.ndarray, train_y: np.ndarray, season_length: int = 1) -> float:
    """Root Mean Squared Scaled Error -- the M5 competition's headline point-forecast
    metric (there, weighted into WRMSSE across a sales hierarchy; unweighted here
    since we're aggregating independent series, not a hierarchy)."""
    return float(np.sqrt(np.mean((actual - forecast) ** 2) / naive_seasonal_scale_sq(train_y, season_length)))


def scaled_pinball_loss(
    actual: np.ndarray, quantiles: dict[float, np.ndarray], train_y: np.ndarray, season_length: int = 1
) -> float:
    """Average pinball loss scaled by the same in-sample seasonal-naive denominator
    as MASE -- M4's uncertainty-track approach (there generalized per-interval as
    MSIS), extending the "safe to average across series" property to quantile
    forecasts, not just point forecasts."""
    return mean_pinball_loss(actual, quantiles) / naive_seasonal_scale(train_y, season_length)


def scaled_interval_width(
    lower: np.ndarray, upper: np.ndarray, train_y: np.ndarray, season_length: int = 1
) -> float:
    """The 80% interval's width (see interval_width), scaled by the same
    in-sample seasonal-naive denominator as MASE/scaled pinball -- unlike plain
    '80% width' (in the series' own units, not comparable across series of
    different scale, exactly like MAE/RMSE), this is safe to average across a
    mixed set of series. Reads as "the band is N times as wide as the series'
    typical single-period change" -- e.g. 2.0 means an 80% band roughly twice
    the size of a normal period-to-period move. Purely a sharpness measure
    (how hedged the uncertainty is), independent of whether that band is well
    calibrated -- always read alongside '80% coverage %', which scaled pinball
    (calibration + sharpness combined into one number) doesn't separate out."""
    return interval_width(lower, upper) / naive_seasonal_scale(train_y, season_length)


# Column-name -> tooltip text, for surfacing in the UI (st.dataframe column_config help=).
METRIC_INFO: dict[str, str] = {
    "MASE": (
        "Mean Absolute Scaled Error: MAE divided by the in-sample error of a ONE-STEP seasonal-naive "
        "forecast (Hyndman & Koehler 2006) -- the M4 competition's headline metric, deliberately scaled "
        "against a single fixed benchmark rather than a different one per horizon. <1 means smaller error "
        "than the series' typical single-period change; >1 means larger. This is NOT the same as 'beats a "
        "naive forecast at this holdout's horizon' -- a naive forecast's own accuracy degrades the further "
        "out it's extrapolated, so even SeasonalNaive itself usually scores >1 here on a multi-step holdout. "
        "Strength: scale-free (safe to average across series of very different magnitude, unlike MAE/RMSE) "
        "and well-defined even at actual=0 (unlike MAPE). Weakness: needs enough in-sample history to "
        "estimate a stable naive-error scale; noisy on very short series."
    ),
    "RMSSE": (
        "Root Mean Squared Scaled Error: like MASE but with squared errors, and the same one-step-naive "
        "scaling caveat -- the M5 competition's headline metric (there weighted into WRMSSE across a sales "
        "hierarchy). Strength: same scale-free safety as MASE, but penalizes large misses more heavily. "
        "Weakness: same as RMSE -- more sensitive to outliers/noise than an absolute-error metric."
    ),
    "scaled pinball": (
        "The same thing as 'avg pinball' (see that column for how the penalty works), just divided by "
        "the same yardstick MASE uses -- the series' typical single-period change -- so it can be "
        "compared across series of very different scale, which is why this column (not avg pinball) is "
        "the one used in the Summary tab's cross-category averages. Lower is better; like avg pinball, "
        "it only means something as a comparison between models on the same series, not as a number to "
        "read in isolation."
    ),
    "MAE": (
        "Mean Absolute Error: average |actual - forecast|, in the series' own units. "
        "Strength: easy to interpret directly against the data. Weakness: not comparable across "
        "series of different scale, and treats all errors linearly (a miss of 20 counts 2x a miss of 10)."
    ),
    "MAPE %": (
        "Mean Absolute Percentage Error: average |actual - forecast| / |actual|, as a %. "
        "Strength: scale-free, so comparable across series. Weaknesses: explodes or is undefined near "
        "actual=0 (a problem for intermittent-demand series), and it penalizes under-forecasts more "
        "harshly than over-forecasts (asymmetric)."
    ),
    "sMAPE %": (
        "Symmetric MAPE: |actual - forecast| / ((|actual| + |forecast|) / 2), as a %. "
        "Strength: bounded and more symmetric than MAPE, more stable near zero. Weakness: still "
        "distorted when both actual and forecast are near zero, and it's less intuitive to explain "
        "than plain MAPE."
    ),
    "RMSE": (
        "Root Mean Squared Error: sqrt(average (actual - forecast)^2), in the series' own units. "
        "Strength: penalizes large misses more than small ones, useful when big errors are "
        "disproportionately costly. Weakness: sensitive to outliers/noise, and like MAE not "
        "comparable across series of different scale."
    ),
    "80% coverage %": (
        "Empirical coverage: % of actual holdout values that fell inside the forecast's 10th-90th "
        "percentile band. Strength: directly checks whether the quantiles are calibrated -- should be "
        "close to 80% if the model's uncertainty estimate is honest. Weakness: needs many holdout "
        "points to be statistically reliable; a single short holdout gives a noisy estimate -- see the "
        "'80% coverage SE %' column next to it for exactly how noisy."
    ),
    "80% coverage SE %": (
        "Standard error of the coverage % to its left, treating each holdout point as an independent "
        "'inside the band or not' coin flip (sqrt(p(1-p)/n)). Use it to judge whether a coverage number "
        "is trustworthy: e.g. 78% coverage from only 10 holdout points has an SE around 13 percentage "
        "points, so the true rate could plausibly be anywhere from the mid-50s to the mid-90s -- that "
        "coverage reading is nearly uninformative. The same 78% from 200 points has an SE under 3 points, "
        "a much tighter and more trustworthy estimate. Weakness: this treats holdout points as "
        "independent, which isn't quite true for a time series (nearby points' errors are correlated), "
        "so it understates the true uncertainty somewhat -- treat it as a lower bound on the noise, not "
        "an exact figure."
    ),
    "80% width": (
        "Average width of the 10th-90th percentile forecast band, in the series' own units. "
        "Strength: shows how confident (narrow) or hedged (wide) the model's uncertainty is. "
        "Weakness: width alone says nothing about calibration -- a band can be narrow and still wrong, "
        "or wide and still miss; always read alongside coverage %. Not comparable across series of "
        "different scale (like MAE/RMSE) -- see 'scaled width' for that."
    ),
    "scaled width": (
        "The same thing as '80% width', divided by the same yardstick MASE uses -- the series' typical "
        "single-period change -- so it's comparable across series of very different scale, which is why "
        "this column (not '80% width') is the one used in the Summary tab's cross-category averages. "
        "Reads as a multiple: 2.0 means the band is about twice as wide as a normal period-to-period "
        "move. Strength: scale-free sharpness measure. Weakness: still says nothing about calibration on "
        "its own -- always read alongside '80% coverage %'; a model can score a great (narrow) scaled "
        "width by simply being overconfident."
    ),
    "avg pinball": (
        "Measures the quality of the whole uncertainty band, not just the point forecast. For each "
        "percentile level (10th, 20th, ... 90th), the forecast at that level is penalized more heavily "
        "for missing on one side than the other, in proportion to the level itself -- e.g. at the 90th "
        "percentile, an actual value that ends up ABOVE the forecast is penalized 9x more heavily than "
        "one that ends up below it. That lopsided penalty is what forces an honest answer: to minimize "
        "it, the model has to set its 90th percentile high enough that only ~10% of actuals exceed it, "
        "not just pick a safely-wide number. This is averaged across all 9 percentile levels into one "
        "score (lower is better). Strength: jointly rewards accuracy and calibration in a single number, "
        "unlike point-only metrics. Weakness: no natural unit, %, or 'good' threshold to compare against -- "
        "it's only meaningful for comparing models against each other on the same series, and it's an "
        "average so it can mask miscalibration at one specific percentile level."
    ),
}
