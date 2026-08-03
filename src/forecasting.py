"""TimesFM 2.5 wrapper: point/quantile forecasting, covariate-driven (XReg)
forecasting, and perturbation-based elasticity estimation.

API details (method signatures, ForecastConfig fields, quantile ordering) were
verified directly against the installed `timesfm` 2.0.2 source rather than
assumed from docs, since the package's public API differs from some online
examples.
"""

from __future__ import annotations

import time

import numpy as np
import timesfm

REPO_ID = "google/timesfm-2.5-200m-pytorch"

# TimesFM_2p5_200M_Definition.quantiles -- index 0 of the quantile output is the
# mean, indices 1-9 correspond to these deciles in order.
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

MAX_CONTEXT = 1024
MAX_HORIZON = 256


def load_model() -> "timesfm.TimesFM_2p5_200M_torch":
    """Loads and compiles the TimesFM 2.5 (200M, torch) model.

    Compiled once with return_backcast=True so the same instance serves both
    plain forecast() calls and forecast_with_covariates() calls (the latter
    requires return_backcast=True at compile time).
    """
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(REPO_ID)
    model.compile(
        timesfm.ForecastConfig(
            max_context=MAX_CONTEXT,
            max_horizon=MAX_HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            fix_quantile_crossing=True,
            return_backcast=True,
        )
    )
    return model


def _quantile_dict(quantile_forecast: np.ndarray) -> dict:
    """quantile_forecast: (horizon, 10) -> {"mean": arr, 0.1: arr, ..., 0.9: arr}."""
    out = {"mean": np.asarray(quantile_forecast[:, 0])}
    for i, q in enumerate(QUANTILE_LEVELS, start=1):
        out[q] = np.asarray(quantile_forecast[:, i])
    return out


def forecast_series(model, context: np.ndarray, horizon: int) -> dict:
    """Plain (no covariates) point + quantile forecast for a single series.

    The model is compiled with return_backcast=True (required for covariate
    support), which makes forecast() return the backcast (reconstruction over
    the padded context) concatenated *before* the horizon-length forecast --
    so the result must be trimmed to its last `horizon` entries.
    """
    t0 = time.perf_counter()
    point, quantiles = model.forecast(horizon=horizon, inputs=[np.asarray(context, dtype=float)])
    point = np.asarray(point[0])[-horizon:]
    quantiles = np.asarray(quantiles[0])[-horizon:]
    calc_ms = (time.perf_counter() - t0) * 1000
    return {"point": point, "quantiles": _quantile_dict(quantiles), "calc_ms": calc_ms}


def forecast_with_covariates(
    model,
    context: np.ndarray,
    horizon: int,
    dynamic_numerical: dict[str, np.ndarray] | None = None,
    dynamic_categorical: dict[str, np.ndarray] | None = None,
    static_numerical: dict[str, float] | None = None,
    static_categorical: dict[str, str] | None = None,
    xreg_mode: str = "xreg + timesfm",
    ridge: float = 0.0,
) -> dict:
    """Covariate-driven forecast for a single series.

    `context` is the historical target window. Each array in
    dynamic_numerical/dynamic_categorical must span the *full* context+horizon
    period (i.e. len(context) + horizon), since future covariate values must
    be known to forecast with them.
    """
    dyn_num = {k: [np.asarray(v, dtype=float)] for k, v in (dynamic_numerical or {}).items()}
    dyn_cat = {k: [list(v)] for k, v in (dynamic_categorical or {}).items()}
    stat_num = {k: [v] for k, v in (static_numerical or {}).items()}
    stat_cat = {k: [v] for k, v in (static_categorical or {}).items()}

    t0 = time.perf_counter()
    point_outputs, quantile_outputs = model.forecast_with_covariates(
        inputs=[np.asarray(context, dtype=float)],
        dynamic_numerical_covariates=dyn_num or None,
        dynamic_categorical_covariates=dyn_cat or None,
        static_numerical_covariates=stat_num or None,
        static_categorical_covariates=stat_cat or None,
        xreg_mode=xreg_mode,
        normalize_xreg_target_per_input=True,
        ridge=ridge,
    )
    point = np.asarray(point_outputs[0])[:horizon]
    quantiles_raw = np.asarray(quantile_outputs[0])[:horizon]
    calc_ms = (time.perf_counter() - t0) * 1000
    return {"point": point, "quantiles": _quantile_dict(quantiles_raw), "calc_ms": calc_ms}


def estimate_price_elasticity(
    model,
    context: np.ndarray,
    horizon: int,
    price_context: np.ndarray,
    price_horizon: np.ndarray,
    other_dynamic_numerical: dict[str, np.ndarray] | None = None,
    static_numerical: dict | None = None,
    static_categorical: dict | None = None,
    pct_bump: float = 0.05,
    xreg_mode: str = "xreg + timesfm",
) -> dict:
    """Estimates price elasticity of demand by perturbing the *horizon-period*
    price covariate by +pct_bump and re-forecasting, holding history and all
    other covariates fixed.

    This is a model-agnostic finite-difference (perturbation) estimate:
    elasticity = %change in mean forecast demand / %change in price.
    It does not depend on inspecting the internal XReg regression coefficients,
    so it works the same way regardless of xreg_mode.
    """

    def run(price_h: np.ndarray) -> dict:
        dyn_num = dict(other_dynamic_numerical or {})
        dyn_num["price"] = np.concatenate([np.asarray(price_context, dtype=float), price_h])
        return forecast_with_covariates(
            model,
            context,
            horizon,
            dynamic_numerical=dyn_num,
            static_numerical=static_numerical,
            static_categorical=static_categorical,
            xreg_mode=xreg_mode,
        )

    price_horizon = np.asarray(price_horizon, dtype=float)
    base = run(price_horizon)
    bumped_price = price_horizon * (1 + pct_bump)
    bumped = run(bumped_price)

    base_q = float(np.mean(base["point"]))
    bumped_q = float(np.mean(bumped["point"]))
    pct_q_change = (bumped_q - base_q) / base_q if base_q != 0 else float("nan")
    elasticity = pct_q_change / pct_bump

    return {
        "base_forecast": base,
        "bumped_forecast": bumped,
        "base_mean_demand": base_q,
        "bumped_mean_demand": bumped_q,
        "pct_price_bump": pct_bump,
        "pct_demand_change": pct_q_change,
        "elasticity": elasticity,
    }
