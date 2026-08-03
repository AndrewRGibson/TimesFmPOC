"""Computes the raw per-(series, model) result rows behind the Summary tab.

Factored out of app.py so the exact same logic can run in two places that
must never drift apart: the standalone precompute script (scripts/
precompute_summary.py, run offline, no Streamlit involved) and, if ever
needed, a live re-run inside the app. This module has no Streamlit
dependency at all -- it's plain functions over the same baselines/
forecasting/metrics/synthetic modules the rest of the app uses.
"""

from __future__ import annotations

import baselines
import forecasting
import metrics
import synthetic


def sample_series_ids(per_category: int) -> list[str]:
    """Evenly spread a sample of `per_category` series ids across each
    category's full list, rather than always taking the first N (which would
    bias the sample toward whatever scale/trend/freq combo happens to sort
    first)."""
    sample: list[str] = []
    for cat in synthetic.list_categories():
        ids = synthetic.series_ids_in_category(cat)
        step = max(1, len(ids) // per_category)
        sample += [ids[i] for i in range(0, len(ids), step)][:per_category]
    return sample


def compute_summary_rows(
    model, sample_ids: list[str], model_choices: list[str], holdout_frac: float,
    progress_callback=None,
) -> list[dict]:
    """Runs TimesFM + the requested classical baselines on every series in
    sample_ids, and returns one row per (series, model) with every metric the
    Summary tab displays -- the same computation tab_summary used to do
    inline, now shared with the precompute script.

    progress_callback(i, n, series_name), if given, is called before each
    series is computed -- lets a caller (e.g. a CLI script) print progress
    without this module knowing anything about how progress is reported.
    """
    catalog = synthetic.get_catalog()
    rows: list[dict] = []
    for i, sid in enumerate(sample_ids):
        spec = catalog[sid]
        if progress_callback:
            progress_callback(i, len(sample_ids), spec.name)

        n = len(spec.df)
        holdout_len = max(1, min(n - 1, forecasting.MAX_HORIZON - 1, round(n * holdout_frac)))
        context_len = max(1, n - holdout_len)
        train_df = spec.df.iloc[: n - holdout_len]
        if context_len < len(train_df):
            train_df = train_df.iloc[-context_len:]
        holdout_df = spec.df.iloc[n - holdout_len :]
        train_y = train_df["y"].to_numpy()
        holdout_y = holdout_df["y"].to_numpy()

        tfm_res = forecasting.forecast_series(model, train_y, len(holdout_df))
        base_res = (
            baselines.run_baselines(train_df[["ds", "y"]], len(holdout_df), model_choices, freq=spec.freq)
            if model_choices
            else {}
        )

        season_length = baselines.DEFAULT_SEASON_LENGTH.get(spec.freq, 1)
        all_results = {"TimesFM": tfm_res, **base_res}
        for model_name, res in all_results.items():
            if res.get("error"):
                continue
            point = res["point"]
            q = res["quantiles"]
            # Scale-free metrics only -- MAE/RMSE/MAPE/sMAPE are deliberately excluded
            # (see the Summary tab's explanation): they aren't meaningful once averaged
            # across series of very different scale and sparsity.
            row = {
                "category": spec.category,
                "model": model_name,
                "n_holdout": len(holdout_y),
                "MASE": metrics.mase(holdout_y, point, train_y, season_length),
                "RMSSE": metrics.rmsse(holdout_y, point, train_y, season_length),
            }
            if 0.1 in q and 0.9 in q:
                row["80% coverage %"] = metrics.coverage(holdout_y, q[0.1], q[0.9])
                row["scaled width"] = metrics.scaled_interval_width(q[0.1], q[0.9], train_y, season_length)
            quantile_only = {k: v for k, v in q.items() if isinstance(k, float)}
            if quantile_only:
                row["scaled pinball"] = metrics.scaled_pinball_loss(holdout_y, quantile_only, train_y, season_length)
            row["calculation ms"] = res.get("calc_ms")
            rows.append(row)
    return rows
