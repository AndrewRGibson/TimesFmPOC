"""Classical regression diagnostics for the covariate / elasticity workflow.

TimesFM's own in-context XReg regression (see forecasting.py, and
timesfm/utils/xreg_lib.py -- BatchedInContextXRegLinear) fits a ridge/OLS linear
model internally on every call, but its coefficients are never returned by the
public API -- only the resulting blended forecast is. The perturbation-based
elasticity estimate in forecasting.estimate_price_elasticity works around that by
finite-differencing the forecast, but it can't give you a coefficient, a standard
error, a p-value, or an R².

This module fits an independent, fully transparent OLS regression on the same
context-window data and the same covariates passed to TimesFM's XReg, purely as a
diagnostic: something to sanity-check the perturbation estimate against, using
textbook statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _fit_ols(target: np.ndarray, regressor_name: str, regressor: np.ndarray, controls: dict[str, np.ndarray]) -> dict:
    X = pd.DataFrame({regressor_name: regressor})
    for name, values in controls.items():
        X[name] = values
    X = sm.add_constant(X)
    model = sm.OLS(target, X).fit()
    ci = model.conf_int().loc[regressor_name]
    return {
        "coef": float(model.params[regressor_name]),
        "se": float(model.bse[regressor_name]),
        "t": float(model.tvalues[regressor_name]),
        "p": float(model.pvalues[regressor_name]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "n_obs": int(model.nobs),
        "all_params": model.params.to_dict(),
        "all_pvalues": model.pvalues.to_dict(),
    }


def fit_price_regression(y: np.ndarray, price: np.ndarray, controls: dict[str, np.ndarray] | None = None) -> dict:
    """Level-level OLS: y ~ price + controls.

    The raw coefficient is in "units of y per dollar of price" (not an
    elasticity). `point_elasticity` rescales it to a price elasticity of demand
    at the sample means (beta * mean(price) / mean(y)) -- the same convention
    used to define this POC's synthetic ground-truth elasticity
    (see synthetic.make_retail_covariates), so it's the directly comparable number.
    """
    result = _fit_ols(y, "price", price, controls or {})
    mean_price, mean_y = float(np.mean(price)), float(np.mean(y))
    result["point_elasticity"] = result["coef"] * mean_price / mean_y if mean_y else float("nan")
    return result


def fit_loglog_elasticity(y: np.ndarray, price: np.ndarray, controls: dict[str, np.ndarray] | None = None) -> dict:
    """Log-log OLS: log(y) ~ log(price) + controls.

    The textbook specification for price elasticity of demand: the price
    coefficient here already *is* the elasticity (% change in y per % change in
    price), with no rescaling needed -- unlike the level-level specification.
    """
    log_y = np.log(np.maximum(y, 1e-6))
    log_price = np.log(np.maximum(price, 1e-6))
    result = _fit_ols(log_y, "log_price", log_price, controls or {})
    result["elasticity"] = result["coef"]
    return result
