"""TimesFM 2.5 forecasting POC -- Streamlit app.

Six workspaces:
  1. Simple Time-Series      -- "Standard" series, holdout, TimesFM quantile fan vs classical baselines
  2. Covariates & Elasticity -- retail series with price/promo/holiday XReg covariates
  3. Hard-to-Forecast Gallery-- short/level-shift/overlapping-cycle/intermittent/volatility/growth-decay series
  4. Summary                 -- M4/M5-style scale-free accuracy aggregated across every category
  5. Upload Your Own Data    -- run the same pipeline on a user-supplied CSV
  6. Changelog               -- renders CHANGELOG.md
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Work around a reproducible Streamlit/PyTorch crash (segfault, not just a caught
# exception): Streamlit's module-path introspection -- used by both the file
# watcher AND the @st.cache_resource/@st.cache_data dependency hashing, so
# `--server.fileWatcherType none` alone does not avoid it -- calls
# `list(torch.classes.__path__._path)`. `torch.classes.__path__` is a special
# proxy (not a real list); accessing it invokes torch's custom-class
# `__getattr__`, which raises a RuntimeError across a pybind11 boundary.
# Streamlit nominally catches and logs that RuntimeError, but the C++ exception
# unwind leaves the process in a state that reproducibly segfaults shortly
# after -- confirmed via repeated crashes in both `AppTest` and the live server
# this project hit during development. Patching `__path__` to a plain empty
# list *before* Streamlit (or anything using its caching) ever runs means the
# buggy `__getattr__` path is never hit at all, regardless of which subsystem
# would have triggered it. Must happen before `import streamlit`.
try:
    import torch

    torch.classes.__path__ = []
except ImportError:
    pass

import numpy as np
import pandas as pd
import streamlit as st

import baselines
import diagnostics
import forecasting
import metrics
import plotting
import synthetic

st.set_page_config(page_title="TimesFM POC", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources / expensive calls
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading TimesFM 2.5 (200M) -- first run downloads ~800MB from Hugging Face...")
def get_model():
    return forecasting.load_model()


def _train_holdout(series_id: str, holdout_len: int, context_len: int):
    spec = synthetic.get_series(series_id)
    df = spec.df
    n = len(df)
    holdout_len = min(holdout_len, n - 1)
    train_df = df.iloc[: n - holdout_len]
    holdout_df = df.iloc[n - holdout_len : n]
    if context_len < len(train_df):
        train_df = train_df.iloc[-context_len:]
    return spec, train_df, holdout_df


@st.cache_data(show_spinner="Running TimesFM...")
def cached_tfm_forecast(_model, series_id: str, holdout_len: int, context_len: int) -> dict:
    _spec, train_df, holdout_df = _train_holdout(series_id, holdout_len, context_len)
    return forecasting.forecast_series(_model, train_df["y"].to_numpy(), len(holdout_df))


@st.cache_data(show_spinner="Running classical baselines...")
def cached_baselines(series_id: str, holdout_len: int, context_len: int, model_names: tuple[str, ...]) -> dict:
    spec, train_df, holdout_df = _train_holdout(series_id, holdout_len, context_len)
    return baselines.run_baselines(train_df[["ds", "y"]], len(holdout_df), list(model_names), freq=spec.freq)


@st.cache_data(show_spinner="Running TimesFM with covariates...")
def cached_tfm_covariate_forecast(_model, series_id: str, holdout_len: int, xreg_mode: str) -> dict:
    spec = synthetic.get_series(series_id)
    _spec, train_df, holdout_df = _train_holdout(series_id, holdout_len, len(spec.df))
    h = len(holdout_df)
    full_df = spec.df.iloc[: len(train_df) + h]
    dyn_num = {c: full_df[c].to_numpy() for c in spec.covariate_cols}
    static_cat = dict(spec.static_covariates)
    return forecasting.forecast_with_covariates(
        _model,
        train_df["y"].to_numpy(),
        h,
        dynamic_numerical=dyn_num,
        static_categorical=static_cat,
        xreg_mode=xreg_mode,
    )


@st.cache_data(show_spinner="Estimating elasticity (2 forecast passes)...")
def cached_elasticity(_model, series_id: str, holdout_len: int, xreg_mode: str, pct_bump: float) -> dict:
    spec = synthetic.get_series(series_id)
    _spec, train_df, holdout_df = _train_holdout(series_id, holdout_len, len(spec.df))
    h = len(holdout_df)
    ctx_len = len(train_df)
    price_context = spec.df["price"].to_numpy()[:ctx_len]
    price_horizon = spec.df["price"].to_numpy()[ctx_len : ctx_len + h]
    other = {
        c: spec.df[c].to_numpy()[: ctx_len + h] for c in spec.covariate_cols if c != "price"
    }
    return forecasting.estimate_price_elasticity(
        _model,
        train_df["y"].to_numpy(),
        h,
        price_context=price_context,
        price_horizon=price_horizon,
        other_dynamic_numerical=other,
        static_categorical=dict(spec.static_covariates),
        pct_bump=pct_bump,
        xreg_mode=xreg_mode,
    )


@st.cache_data(show_spinner=False)
def cached_regression_diagnostics(series_id: str, holdout_len: int, include_trend: bool) -> dict:
    spec = synthetic.get_series(series_id)
    _spec, train_df, _holdout_df = _train_holdout(series_id, holdout_len, len(spec.df))
    y = train_df["y"].to_numpy()
    price = train_df["price"].to_numpy()
    controls = {c: train_df[c].to_numpy() for c in spec.covariate_cols if c != "price"}
    t = np.arange(len(train_df), dtype=float)
    if include_trend:
        controls["t"] = t
    return {
        "level": diagnostics.fit_price_regression(y, price, controls),
        "loglog": diagnostics.fit_loglog_elasticity(y, price, controls),
        "corr_price_t": float(np.corrcoef(price, t)[0, 1]) if len(t) > 1 else 0.0,
        "n": len(train_df),
    }


# Upload-path caching: unlike the synthetic-catalog functions above (keyed by
# series_id, looked up via synthetic.get_series), uploaded data has no catalog
# entry, so these take the raw arrays/frames directly -- Streamlit's cache_data
# hashes numpy arrays and DataFrames natively, so this works the same way.
@st.cache_data(show_spinner="Running TimesFM on your data...")
def cached_tfm_forecast_upload(_model, train_y: np.ndarray, horizon: int) -> dict:
    return forecasting.forecast_series(_model, train_y, horizon)


@st.cache_data(show_spinner="Running classical baselines on your data...")
def cached_baselines_upload(train_df: pd.DataFrame, horizon: int, model_names: tuple[str, ...], freq: str) -> dict:
    return baselines.run_baselines(train_df[["ds", "y"]], horizon, list(model_names), freq=freq)


@st.cache_data(show_spinner="Running TimesFM with your covariates...")
def cached_tfm_covariate_forecast_upload(_model, train_y: np.ndarray, horizon: int, dyn_num: dict[str, np.ndarray]) -> dict:
    return forecasting.forecast_with_covariates(_model, train_y, horizon, dynamic_numerical=dyn_num)


# ---------------------------------------------------------------------------
# Shared UI helpers
# ---------------------------------------------------------------------------


def series_picker(
    key_prefix: str,
    categories: list[str] | None = None,
    default_category: str | None = None,
    show_category_selector: bool = True,
):
    catalog = synthetic.get_catalog()
    all_categories = synthetic.list_categories()
    cats = categories or all_categories

    if show_category_selector:
        cat = st.selectbox(
            "Category", cats, index=cats.index(default_category) if default_category in cats else 0,
            key=f"{key_prefix}_cat",
        )
    else:
        cat = default_category or cats[0]

    ids_in_cat = synthetic.series_ids_in_category(cat)
    labels = {sid: catalog[sid].name for sid in ids_in_cat}
    sel_key = f"{key_prefix}_series_select"
    prev_cat_key = f"{key_prefix}_prev_cat"
    if st.session_state.get(prev_cat_key) != cat:
        st.session_state[sel_key] = ids_in_cat[0]
        st.session_state[prev_cat_key] = cat

    def _step(delta: int):
        cur = st.session_state.get(sel_key, ids_in_cat[0])
        i = ids_in_cat.index(cur) if cur in ids_in_cat else 0
        st.session_state[sel_key] = ids_in_cat[(i + delta) % len(ids_in_cat)]

    nav_l, nav_mid, nav_r = st.columns([1, 5, 1])
    nav_l.button("◀", key=f"{key_prefix}_prev_btn", on_click=_step, args=(-1,), use_container_width=True)
    nav_r.button("▶", key=f"{key_prefix}_next_btn", on_click=_step, args=(1,), use_container_width=True)
    with nav_mid:
        series_id = st.selectbox(
            "Series", ids_in_cat, format_func=lambda sid: labels[sid], key=sel_key, label_visibility="collapsed"
        )

    idx = ids_in_cat.index(series_id)
    spec = catalog[series_id]
    st.caption(f"**{idx + 1} of {len(ids_in_cat)}** in _{cat}_ -- use ◀ / ▶ to flip through them")
    st.caption(spec.description)
    return spec


def _bounded_slider(label: str, lo: int, hi: int, default: int, key: str, help: str | None = None) -> int:
    """A slider guarded against Streamlit's `min_value < max_value` requirement --
    very short series (e.g. an 8-point monthly series) can legitimately collapse
    the valid range to a single value, which a real st.slider call would raise on."""
    lo, hi = int(lo), int(hi)
    if hi <= lo:
        st.caption(f"{label}: **{hi}** (fixed -- not enough data to vary)")
        return hi
    default = int(max(lo, min(hi, default)))
    return st.slider(label, min_value=lo, max_value=hi, value=default, key=key, help=help)


def holdout_context_controls(key_prefix: str, n: int, freq: str, default_holdout_frac: float = 0.15):
    unit = synthetic.FREQ_UNIT[freq]
    min_holdout = 1
    max_holdout = max(min_holdout, min(180, n - 2, forecasting.MAX_HORIZON - 1))
    default_holdout = max(min_holdout, min(max_holdout, round(n * default_holdout_frac)))
    holdout_len = _bounded_slider(f"Holdout length ({unit})", min_holdout, max_holdout, default_holdout, key=f"{key_prefix}_holdout")

    max_context = max(1, n - holdout_len)
    min_context = min(10, max_context)
    context_len = _bounded_slider(
        f"Context length fed to model ({unit})", min_context, max_context, max_context,
        key=f"{key_prefix}_context",
        help="Shrink this to test the model's few-shot behavior with less history.",
    )
    return holdout_len, context_len


def metrics_table(holdout_y: np.ndarray, train_y: np.ndarray, season_length: int, results: dict[str, dict]) -> pd.DataFrame:
    """Per-series accuracy table. Leads with MASE/RMSSE (scale-free, comparable to a
    seasonal-naive benchmark) then the classic raw-unit/percentage metrics -- all of
    which remain meaningful for a single series in isolation, unlike when averaged
    across series of different scale (see the Summary tab for that case)."""
    rows = []
    for name, res in results.items():
        if res.get("error"):
            rows.append({"model": name, "error": res["error"]})
            continue
        point = res["point"]
        q = res["quantiles"]
        row = {
            "model": name,
            "MASE": metrics.mase(holdout_y, point, train_y, season_length),
            "RMSSE": metrics.rmsse(holdout_y, point, train_y, season_length),
            "MAE": metrics.mae(holdout_y, point),
            "MAPE %": metrics.mape(holdout_y, point),
            "sMAPE %": metrics.smape(holdout_y, point),
            "RMSE": metrics.rmse(holdout_y, point),
        }
        if 0.1 in q and 0.9 in q:
            row["80% coverage %"] = metrics.coverage(holdout_y, q[0.1], q[0.9])
            row["80% coverage SE %"] = metrics.coverage_se(holdout_y, q[0.1], q[0.9])
            row["80% width"] = metrics.interval_width(q[0.1], q[0.9])
            row["scaled width"] = metrics.scaled_interval_width(q[0.1], q[0.9], train_y, season_length)
        quantile_only = {k: v for k, v in q.items() if isinstance(k, float)}
        if quantile_only:
            row["avg pinball"] = metrics.mean_pinball_loss(holdout_y, quantile_only)
            row["scaled pinball"] = metrics.scaled_pinball_loss(holdout_y, quantile_only, train_y, season_length)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def render_metrics_table(holdout_y: np.ndarray, train_y: np.ndarray, season_length: int, results: dict[str, dict]) -> None:
    table = metrics_table(holdout_y, train_y, season_length, results)
    col_config = {
        col: st.column_config.NumberColumn(col, help=metrics.METRIC_INFO[col], format="%.2f")
        for col in table.columns
        if col in metrics.METRIC_INFO
    }
    st.dataframe(table, use_container_width=True, column_config=col_config)


def render_summary_stats(y: np.ndarray, label: str = "Actual data") -> None:
    s = pd.Series(y)
    stats = pd.DataFrame(
        [
            {
                "n": int(s.count()),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "25%": s.quantile(0.25),
                "median": s.median(),
                "75%": s.quantile(0.75),
                "max": s.max(),
            }
        ]
    )
    st.caption(f"**Summary statistics -- {label}**")
    st.dataframe(stats.style.format(precision=2), use_container_width=True, hide_index=True)


def render_calibration_section(holdout_y: np.ndarray, tfm_res: dict) -> None:
    with st.expander("Are TimesFM's quantiles reasonable? (calibration check)"):
        st.caption(
            "For each nominal quantile level, what fraction of holdout actuals actually fell at or below "
            "the forecast at that level? A well-calibrated model tracks the y=x diagonal -- e.g. ~70% of "
            "actuals should fall at or below the 70th-percentile forecast. Points above the line mean the "
            "model is under-forecasting at that level (actuals exceed it more often than expected); points "
            "below mean it's over-forecasting. **Caveat**: with only a handful of holdout points this "
            "estimate is noisy -- use a longer holdout for a more reliable read."
        )
        q = tfm_res["quantiles"]
        levels = sorted(k for k in q if isinstance(k, float))
        rows = [
            {
                "nominal quantile": lvl,
                "empirical coverage": float(np.mean(holdout_y <= q[lvl])),
                "gap (empirical - nominal)": float(np.mean(holdout_y <= q[lvl])) - lvl,
            }
            for lvl in levels
        ]
        calib_df = pd.DataFrame(rows)
        fig = plotting.calibration_chart(calib_df)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(calib_df.style.format(precision=2), use_container_width=True, hide_index=True)


# Colored-square emoji standing in for each series' actual chart color, so the
# checkbox row below reads like a (clickable) legend rather than a plain list.
# Plotly's own per-chart legend already supports click-to-toggle natively, but
# that's local to one figure and the detail chart has no legend at all (see
# render_fan_chart_pair) -- this is what makes one control affect both charts.
_VISIBILITY_LABELS = {
    "TimesFM (point)": "TimesFM",
    "TimesFM quantile bands": "TimesFM bands",
    "Actual (holdout)": "Actual",
}


_FIXED_SWATCH_COLORS = {
    "History": plotting.COLOR_CONTEXT,
    "TimesFM (point)": plotting.COLOR_TIMESFM,
    "TimesFM quantile bands": plotting.COLOR_TIMESFM,
    "Actual (holdout)": plotting.COLOR_ACTUAL,
}


def _visibility_color(name: str) -> str:
    """The EXACT color that series is actually drawn with on the chart --
    reusing the same lookup _add_fan_traces itself uses (right down to its
    INK_SECONDARY fallback for unrecognized baseline names, e.g. 'TimesFM (no
    covariates)' isn't in BASELINE_COLORS and really is drawn gray, not a
    color made up to look distinct)."""
    if name in _FIXED_SWATCH_COLORS:
        return _FIXED_SWATCH_COLORS[name]
    return plotting.BASELINE_COLORS.get(name, plotting.INK_SECONDARY)


def _visibility_swatch_css(name: str) -> str:
    """Background for the swatch -- a solid bar for series drawn as solid lines
    (History, TimesFM, Actual), or a dotted pattern for baseline models, which
    _add_fan_traces always draws with dash='dot'. Keeps the selector honest
    about line style, not just color."""
    color = _visibility_color(name)
    if name in _FIXED_SWATCH_COLORS:
        return f"background:{color};"
    return f"background-image: repeating-linear-gradient(90deg, {color} 0 4px, transparent 4px 8px);"


def _format_season_length(sl) -> str:
    if isinstance(sl, list):
        return ",".join(str(x) for x in sl)
    return str(sl)


def _baseline_display_label(name: str, season_lengths: dict[str, object] | None) -> str:
    """Appends the actual lag/period a model fit with, e.g. 'SeasonalNaive
    (lag 365)' -- otherwise this is invisible in the UI, which is exactly what
    let an earlier hardcoded-lag mismatch (see detect_season_length in
    baselines.py) go unnoticed: the detected lag needs to be visible at a
    glance, not just documented in a tooltip."""
    sl = (season_lengths or {}).get(name)
    if sl is None:
        return name
    return f"{name} (lag {_format_season_length(sl)})"


def series_visibility_selector(
    key_prefix: str, baseline_names: list[str], season_lengths: dict[str, object] | None = None
) -> set[str]:
    """A row of checkboxes, one per series, each with a small color swatch in
    the series' EXACT chart color directly above it -- the same component
    doing double duty as legend and control, rather than a separate list with
    icons that only approximated the real colors (and could be flat wrong, as
    "TimesFM (no covariates)" was). Controls BOTH the overview and detail
    chart in a pair (they're built from the same filtered selection), without
    needing to re-run any forecast."""
    options = ["History", "TimesFM (point)", "TimesFM quantile bands", *baseline_names, "Actual (holdout)"]
    cols = st.columns(len(options))
    selected = set()
    for col, name in zip(cols, options):
        swatch_css = _visibility_swatch_css(name)
        label = _VISIBILITY_LABELS.get(name) or _baseline_display_label(name, season_lengths)
        with col:
            st.markdown(
                f'<div style="height:5px;border-radius:2px;margin-bottom:3px;{swatch_css}"></div>',
                unsafe_allow_html=True,
            )
            if st.checkbox(label, value=True, key=f"{key_prefix}_vis_{name}"):
                selected.add(name)
    return selected


_MODEL_NOTES = {
    "AutoETS": "can be slow when the detected season length is long",
}


def baseline_model_selector(key_prefix: str, default: list[str]) -> list[str]:
    """Checkbox list rather than a multiselect dropdown, so every available
    baseline -- and any that are notably slower to fit -- is visible at a
    glance instead of hidden behind a click. Each checkbox's (?) tooltip
    explains what that model actually does and doesn't account for (e.g.
    SeasonalNaive has no idea trend exists) -- not just its name."""
    st.caption("Baseline models")
    selected = []
    for name in baselines.MODEL_CHOICES:
        note = _MODEL_NOTES.get(name)
        label = f"{name} ({note})" if note else name
        if st.checkbox(label, value=name in default, key=f"{key_prefix}_model_{name}", help=baselines.MODEL_INFO.get(name)):
            selected.append(name)
    return selected


def _add_zoom_rect(overview_fig, detail_fig, row: int | None = None, col: int | None = None) -> None:
    """Outlines, on the overview chart, the exact x/y region the paired detail
    chart is zoomed into -- read directly off the already-computed detail
    figure's axis ranges rather than recomputing them."""
    x_range = detail_fig.layout.xaxis.range
    y_range = detail_fig.layout.yaxis.range
    if not x_range or not y_range:
        return
    overview_fig.add_shape(
        type="rect", x0=x_range[0], x1=x_range[1], y0=y_range[0], y1=y_range[1],
        line=dict(color="rgba(148, 148, 148, 0.55)", width=1), fillcolor="rgba(0,0,0,0)", layer="above",
        row=row, col=col,
    )


_COMPACT_TOP_MARGIN = dict(t=50)


def render_fan_chart_pair(
    title: str,
    key_prefix: str,
    baseline_names: list[str] | None = None,
    season_lengths: dict[str, object] | None = None,
    **kwargs,
) -> None:
    """Overview chart (full history, zero-anchored) next to a detail chart
    (tight x/y range around just the forecast window), with a light outline on
    the overview marking exactly what region the detail chart is zoomed into.
    Neither chart draws Plotly's own legend -- the color-matched checkbox row
    from series_visibility_selector above them already IS the legend (and the
    control), in the exact same colors, so a second native legend would just
    be redundant. Both figures get the same reduced top margin (freed up now
    that neither needs room for a legend), which is also what keeps their plot
    areas the same pixel height side by side."""
    visible = series_visibility_selector(key_prefix, baseline_names or [], season_lengths)

    detail_fig = plotting.fan_chart(title="Detail", zoom=True, visible=visible, **kwargs)
    detail_fig.update_layout(showlegend=False, margin=_COMPACT_TOP_MARGIN)

    overview_fig = plotting.fan_chart(title=title, zoom=False, visible=visible, **kwargs)
    overview_fig.update_layout(showlegend=False, margin=_COMPACT_TOP_MARGIN)
    _add_zoom_rect(overview_fig, detail_fig)

    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.plotly_chart(overview_fig, use_container_width=True)
    with col2:
        st.plotly_chart(detail_fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 1: Fit & Forecast
# ---------------------------------------------------------------------------


def tab_fit_forecast(model):
    st.caption(
        "General-purpose forecasting on the 'Standard' series (trend + seasonality, easy baseline "
        "cases). Covariate series and the deliberately hard-to-forecast categories have their own "
        "dedicated tabs -- this one focuses on the everyday case."
    )
    left, right = st.columns([1, 3], gap="large")

    with left, st.container(border=True):
        spec = series_picker("t1", categories=["Standard"], default_category="Standard", show_category_selector=False)
        n = len(spec.df)
        holdout_len, context_len = holdout_context_controls("t1", n, spec.freq)
        model_choices = baseline_model_selector("t1", default=["SeasonalNaive", "AutoETS"])

    with right:
        tfm_res = cached_tfm_forecast(model, spec.series_id, holdout_len, context_len)
        base_res = cached_baselines(spec.series_id, holdout_len, context_len, tuple(model_choices)) if model_choices else {}

        _spec, train_df, holdout_df = _train_holdout(spec.series_id, holdout_len, context_len)
        render_fan_chart_pair(
            title=f"{spec.name} -- {holdout_len} {synthetic.FREQ_UNIT[spec.freq]} holdout",
            key_prefix="t1", baseline_names=list(base_res.keys()),
            season_lengths={name: res.get("season_length") for name, res in base_res.items()},
            context_dates=train_df["ds"], context_y=train_df["y"].to_numpy(), forecast_dates=holdout_df["ds"],
            tfm_result=tfm_res, holdout_y=holdout_df["y"].to_numpy(), baseline_results=base_res,
        )

        all_results = {"TimesFM": tfm_res, **base_res}
        st.markdown("**Accuracy on holdout**")
        season_length = baselines.DEFAULT_SEASON_LENGTH.get(spec.freq, 1)
        render_metrics_table(holdout_df["y"].to_numpy(), train_df["y"].to_numpy(), season_length, all_results)

        render_summary_stats(spec.df["y"].to_numpy(), label=f"{spec.name}, full series")
        render_calibration_section(holdout_df["y"].to_numpy(), tfm_res)


# ---------------------------------------------------------------------------
# Tab 2: Covariates & Elasticity
# ---------------------------------------------------------------------------


def tab_covariates(model):
    st.caption(
        "Retail-style series with price / promotion / holiday covariates, forecast via TimesFM's XReg support. "
        "Elasticity is estimated by perturbing the horizon-period price and re-forecasting (finite-difference), "
        "then checked against the synthetic data's known ground-truth elasticity."
    )
    left, right = st.columns([1, 3], gap="large")

    with left, st.container(border=True):
        spec = series_picker(
            "t2", categories=["Covariates / elasticity"], default_category="Covariates / elasticity",
            show_category_selector=False,
        )
        series_id = spec.series_id
        n = len(spec.df)
        holdout_len, _ = holdout_context_controls("t2", n, spec.freq, default_holdout_frac=0.08)
        xreg_mode = st.radio(
            "XReg mode", ["xreg + timesfm", "timesfm + xreg"], key="t2_mode",
            help="'xreg + timesfm': fit regression on target, TimesFM forecasts residuals. 'timesfm + xreg': TimesFM forecasts first, regression fits residuals.",
        )
        pct_bump = st.slider("Price bump for elasticity test (%)", -20, 20, 5, key="t2_bump") / 100

    with right:
        _spec, train_df, holdout_df = _train_holdout(series_id, holdout_len, len(spec.df))
        window_df = spec.df.iloc[: len(train_df) + holdout_len]

        st.markdown(
            f"**Covariates**: `price` (dollars, numeric) -- store's selling price. "
            f"`promo` (0/1 flag) -- promotional week. `holiday` (0/1 flag) -- holiday period. "
            "All three are known for the forecast horizon (required for XReg) as well as history, "
            "plotted directly below the sales forecast on the same shared x-axis."
        )

        cov_res = cached_tfm_covariate_forecast(model, series_id, holdout_len, xreg_mode)
        plain_res = cached_tfm_forecast(model, series_id, holdout_len, len(train_df))

        cov_chart_kwargs = dict(
            covariate_df=window_df, covariate_cols=spec.covariate_cols,
            context_dates=train_df["ds"], context_y=train_df["y"].to_numpy(), forecast_dates=holdout_df["ds"],
            tfm_result=cov_res, holdout_y=holdout_df["y"].to_numpy(),
            baseline_results={"TimesFM (no covariates)": plain_res},
        )
        cov_visible = series_visibility_selector("t2", ["TimesFM (no covariates)"])
        cov_title = f"{spec.name} -- with vs without covariates"

        cov_detail_fig = plotting.covariate_fan_chart(title="Detail", zoom=True, visible=cov_visible, **cov_chart_kwargs)
        cov_detail_fig.update_layout(showlegend=False, margin=_COMPACT_TOP_MARGIN)

        cov_overview_fig = plotting.covariate_fan_chart(title=cov_title, zoom=False, visible=cov_visible, **cov_chart_kwargs)
        cov_overview_fig.update_layout(showlegend=False, margin=_COMPACT_TOP_MARGIN)
        _add_zoom_rect(cov_overview_fig, cov_detail_fig, row=1, col=1)

        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.plotly_chart(cov_overview_fig, use_container_width=True)
        with col2:
            st.plotly_chart(cov_detail_fig, use_container_width=True)

        with st.expander("Covariate values (table)"):
            st.dataframe(
                window_df[["ds", *spec.covariate_cols]].style.format(precision=3),
                use_container_width=True, hide_index=True,
            )

        st.markdown("**Accuracy on holdout: covariates vs no covariates**")
        season_length = baselines.DEFAULT_SEASON_LENGTH.get(spec.freq, 1)
        render_metrics_table(
            holdout_df["y"].to_numpy(), train_df["y"].to_numpy(), season_length,
            {"TimesFM + covariates": cov_res, "TimesFM (no covariates)": plain_res},
        )
        render_summary_stats(spec.df["y"].to_numpy(), label=f"{spec.name}, full series")

        st.markdown("---")
        st.markdown("**Price elasticity of demand** (perturbation estimate)")
        elas = cached_elasticity(model, series_id, holdout_len, xreg_mode, pct_bump)
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated elasticity", f"{elas['elasticity']:.2f}")
        c2.metric("Ground-truth elasticity", f"{spec.true_price_elasticity:.2f}")
        c3.metric("Demand change from bump", f"{elas['pct_demand_change']*100:+.1f}%")
        elas_fig = plotting.elasticity_chart(
            holdout_df["ds"], elas["base_forecast"]["point"], elas["bumped_forecast"]["point"], pct_bump
        )
        st.plotly_chart(elas_fig, use_container_width=True)

        st.markdown("---")
        st.markdown("**Classical regression diagnostics**")
        st.caption(
            "TimesFM's own in-context XReg regression (see forecasting.py / xreg_lib.py) fits a linear "
            "model internally on every call, but never exposes its coefficients through the public API -- "
            "only the resulting blended forecast. This fits an independent, fully transparent OLS "
            "regression on the same context-window data and the same covariates, purely as a diagnostic: "
            "a coefficient, standard error, t-statistic, p-value, confidence interval, and R² that the "
            "perturbation estimate above can't give you."
        )
        include_trend = st.checkbox(
            "Control for the underlying time trend", value=True, key="t2_trend_control",
            help="Price here is a random walk that can end up spuriously correlated with the series' own "
            "trend just by chance over any given window -- inflating the price coefficient if not "
            "controlled for. Uncheck to see the naive (biased) version for comparison.",
        )
        diag = cached_regression_diagnostics(series_id, holdout_len, include_trend)
        lvl, loglog = diag["level"], diag["loglog"]

        if abs(diag["corr_price_t"]) > 0.3:
            st.warning(
                f"Price and time are correlated at r={diag['corr_price_t']:.2f} over this context window "
                "(pure chance in the random walk) -- without the trend control, the price coefficient "
                "would partly be picking up the trend's effect instead of price's."
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ground-truth elasticity", f"{spec.true_price_elasticity:.2f}")
        c2.metric("Perturbation estimate", f"{elas['elasticity']:.2f}")
        c3.metric("Regression (level-level)", f"{lvl['point_elasticity']:.2f}")
        c4.metric("Regression (log-log)", f"{loglog['elasticity']:.2f}")

        trend_suffix = " + t" if include_trend else ""
        diag_table = pd.DataFrame(
            [
                {
                    "specification": f"Level-level: sales ~ price{trend_suffix}",
                    "coefficient": lvl["coef"],
                    "implied elasticity": lvl["point_elasticity"],
                    "std err": lvl["se"],
                    "t-stat": lvl["t"],
                    "p-value": lvl["p"],
                    "95% CI low": lvl["ci_low"],
                    "95% CI high": lvl["ci_high"],
                    "R²": lvl["r_squared"],
                    "n": lvl["n_obs"],
                },
                {
                    "specification": f"Log-log: log(sales) ~ log(price){trend_suffix}",
                    "coefficient": loglog["coef"],
                    "implied elasticity": loglog["elasticity"],
                    "std err": loglog["se"],
                    "t-stat": loglog["t"],
                    "p-value": loglog["p"],
                    "95% CI low": loglog["ci_low"],
                    "95% CI high": loglog["ci_high"],
                    "R²": loglog["r_squared"],
                    "n": loglog["n_obs"],
                },
            ]
        )
        st.dataframe(diag_table.style.format(precision=3), use_container_width=True, hide_index=True)
        st.caption(
            "Level-level's raw coefficient is in units of sales per dollar of price -- 'implied elasticity' "
            "rescales it at the sample means (beta x mean(price)/mean(sales)), the same convention used to "
            "define this app's synthetic ground-truth elasticity. Log-log's coefficient *is* the elasticity "
            "directly, no rescaling needed."
        )
        with st.expander("Full regression output (all coefficients, both specifications)"):
            st.write("Level-level:", lvl["all_params"])
            st.write("Log-log:", loglog["all_params"])


# ---------------------------------------------------------------------------
# Tab 3: Hard-to-Forecast Gallery
# ---------------------------------------------------------------------------

HARD_CATEGORIES = [
    "Short series",
    "Level shift",
    "Overlapping cycles",
    "Intermittent demand",
    "Volatility regime change",
    "Growth & decay traps",
]


def tab_gallery(model):
    st.caption(
        "Deliberately difficult scenarios: very short history, abrupt structural breaks (red dashed "
        "markers), multiple non-integer seasonal cycles, sparse/bursty demand, sudden volatility "
        "regime changes, and growth/decay traps (red dashed markers) -- apparent exponential growth "
        "that saturates, and rapid decay toward the non-negativity floor. The chart always shows full "
        "history so these events are never cropped out of view."
    )
    left, right = st.columns([1, 3], gap="large")

    with left, st.container(border=True):
        spec = series_picker("t3", categories=HARD_CATEGORIES)
        n = len(spec.df)
        holdout_len, context_len = holdout_context_controls("t3", n, spec.freq, default_holdout_frac=0.25)
        model_choices = baseline_model_selector("t3", default=["SeasonalNaive", "AutoETS"])

    with right:
        tfm_res = cached_tfm_forecast(model, spec.series_id, holdout_len, context_len)
        base_res = cached_baselines(spec.series_id, holdout_len, context_len, tuple(model_choices)) if model_choices else {}

        _spec, train_df, holdout_df = _train_holdout(spec.series_id, holdout_len, context_len)
        event_dates = [spec.df["ds"].iloc[idx] for idx in spec.event_points if idx < len(spec.df)]
        render_fan_chart_pair(
            title=f"{spec.name} -- {holdout_len} {synthetic.FREQ_UNIT[spec.freq]} holdout",
            key_prefix="t3", baseline_names=list(base_res.keys()),
            season_lengths={name: res.get("season_length") for name, res in base_res.items()},
            context_dates=train_df["ds"], context_y=train_df["y"].to_numpy(), forecast_dates=holdout_df["ds"],
            tfm_result=tfm_res, holdout_y=holdout_df["y"].to_numpy(), baseline_results=base_res,
            event_dates=event_dates,
        )

        all_results = {"TimesFM": tfm_res, **base_res}
        st.markdown("**Accuracy on holdout**")
        season_length = baselines.DEFAULT_SEASON_LENGTH.get(spec.freq, 1)
        render_metrics_table(holdout_df["y"].to_numpy(), train_df["y"].to_numpy(), season_length, all_results)

        render_summary_stats(spec.df["y"].to_numpy(), label=f"{spec.name}, full series")
        render_calibration_section(holdout_df["y"].to_numpy(), tfm_res)


# ---------------------------------------------------------------------------
# Tab 4: Summary
# ---------------------------------------------------------------------------


def _sample_series_ids(per_category: int) -> list[str]:
    """Evenly spread a sample of `per_category` series ids across each category's
    full list, rather than always taking the first N (which would bias the sample
    toward whatever scale/trend/freq combo happens to sort first)."""
    sample: list[str] = []
    for cat in synthetic.list_categories():
        ids = synthetic.series_ids_in_category(cat)
        step = max(1, len(ids) // per_category)
        sample += [ids[i] for i in range(0, len(ids), step)][:per_category]
    return sample


def _aggregate_summary(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    """Simple mean per metric, EXCEPT '80% coverage %', which is weighted by
    each row's holdout length (n_holdout). Coverage is an empirical hit rate,
    so pooling by point count -- not one equally-weighted number per series
    regardless of how many holdout points backed it -- is the standard way to
    combine proportions from groups of very different sample sizes: a 4-point
    annual series' coverage estimate is far noisier than a 150-point daily
    one's (see the '80% coverage SE %' column on the per-series tables), and
    shouldn't move the aggregate just as much as the daily series does."""
    def agg(group: pd.DataFrame) -> pd.Series:
        out = {}
        for col in metric_cols:
            valid = group[col].notna()
            if col == "80% coverage %" and valid.any():
                out[col] = np.average(group.loc[valid, col], weights=group.loc[valid, "n_holdout"])
            else:
                out[col] = group.loc[valid, col].mean()
        return pd.Series(out)

    return df.groupby(group_cols, sort=False).apply(agg, include_groups=False).reset_index()


def tab_summary(model):
    st.caption(
        "Runs TimesFM (and optionally classical baselines) across a sample of series from every "
        "category and aggregates accuracy by category and overall. Each series needs its own forecast "
        "pass, so larger samples take longer -- results already computed in other tabs are reused instantly."
    )
    st.markdown(
        """
**Why these particular metrics, and not MAE/RMSE/MAPE?**

This tab averages accuracy across dozens of series that span wildly different scales
(a "Standard" series around 20-30 units vs. a monthly retail series around 500),
frequencies (daily/weekly/monthly/annual), and even sparsity (the "Intermittent
demand" category is mostly zeros). Three of the metrics used elsewhere in this app
break under that mix:

- **MAE and RMSE are in the series' own units.** Averaging them across series of
  different magnitude is dominated by whichever series happens to have the largest
  values -- it doesn't reflect *relative* accuracy at all. A model that's excellent
  on a small-scale series and mediocre on a large-scale one can still show a "good"
  average MAE just because the large series' errors are bigger numbers.
- **MAPE is undefined (or explodes) when the actual value is near zero** -- exactly
  the situation for every observation in the intermittent-demand category, and for
  any series with a near-zero point in its holdout.

So this tab instead uses the approach the **M4** (Makridakis et al., 2020) and
**M5** (Makridakis, Spiliotis & Assimakopoulos, 2022) forecasting competitions
adopted for exactly this reason -- comparing thousands of heterogeneous series on
one scoreboard:

- **MASE** (Mean Absolute Scaled Error) -- M4's headline point-forecast metric.
  It divides the model's MAE by the in-sample MAE of a simple one-step
  seasonal-naive forecast (e.g. "today = same day last week" for daily data).
  That makes it **scale-free** (a MASE of 0.8 means "20% better than seasonal-naive"
  regardless of whether the series runs in the tens or the thousands) and it stays
  well-defined at zero actuals, unlike MAPE.
- **RMSSE** (Root Mean Squared Scaled Error) -- the same idea with squared errors,
  M5's headline point-forecast metric (there weighted into WRMSSE across a sales
  hierarchy; here left unweighted since these are independent series, not a
  hierarchy). It shares MASE's scale-free property but penalizes large misses
  more heavily.
- **80% coverage** is already a percentage, so it's scale-free -- but it's still
  weighted by each series' holdout length here, not averaged one-series-one-vote.
  Coverage is an empirical hit rate, and a 4-point annual series' coverage estimate
  is far noisier than a 150-point daily one's (see "80% coverage SE %" on the
  per-series tables); pooling by point count is the standard way to combine rates
  from groups of very different sample sizes, and keeps a handful of tiny holdouts
  from swinging the aggregate as much as one large one.
- **Scaled pinball loss** applies the same MASE-style scaling to the average
  pinball (quantile) loss -- the spirit of M4's MSIS uncertainty metric -- so
  quantile-forecast quality is comparable across series too.

**Important nuance: the naive benchmark is always the *one-step-ahead* naive
error, even though our holdout is multi-step.** This is the standard M4/M5
definition (Hyndman & Koehler define it that way, deliberately, so the metric
has one stable, computable reference point instead of a different one for every
possible horizon) -- but it means **MASE < 1 does not mean "beats a naive
forecast at this holdout's horizon."** It means "the model's average error per
period is smaller than the series' typical single-period change." A naive
forecast's *own* error grows the further out it's extrapolated (it has no way to
anticipate multi-step drift), so a naive forecast repeated 30 days out will
usually score MASE well above 1 -- **not because MASE is broken, but because
1-step naive error is a much easier bar than 30-step naive accuracy.** You can
see this directly: SeasonalNaive is often included as a baseline model in the
tables below, and its own MASE/RMSSE row typically prints well above 1 for
exactly this reason. Read MASE/RMSSE as "error relative to a fixed, stable
yardstick," not as a literal horizon-matched naive comparison.

The per-series tables in the other three tabs still show MAE/RMSE/MAPE/sMAPE
alongside MASE/RMSSE, since those raw-unit metrics remain perfectly meaningful
when you're looking at one series at a time -- it's only *averaging* them across
heterogeneous series that breaks.
"""
    )
    catalog = synthetic.get_catalog()
    categories = synthetic.list_categories()
    max_per_cat = min(len(synthetic.series_ids_in_category(c)) for c in categories)

    left, right = st.columns([1, 3], gap="large")
    with left, st.container(border=True):
        per_category = st.slider(
            "Series per category (sampled)", 1, max_per_cat, min(3, max_per_cat), key="sum_n",
            help=f"{len(categories)} categories x N series. Series are spread evenly across each "
            f"category's {max_per_cat} instances, not just the first N.",
        )
        model_choices = baseline_model_selector("sum", default=["SeasonalNaive", "AutoETS"])
        holdout_frac = st.slider("Holdout fraction of each series", 0.05, 0.4, 0.15, step=0.05, key="sum_holdout_frac")
        st.caption(f"Will run **{per_category * len(categories)} series** -- may take a few minutes for larger samples.")
        run = st.button("Run summary sweep", type="primary", key="sum_run")

    if not run and "sum_ran" not in st.session_state:
        with right:
            st.info("Configure the sample size and click **Run summary sweep**.")
        return
    st.session_state["sum_ran"] = True

    with right:
        sample_ids = _sample_series_ids(per_category)
        progress = st.progress(0.0, text="Starting sweep...")
        rows = []
        for i, sid in enumerate(sample_ids):
            spec = catalog[sid]
            n = len(spec.df)
            holdout_len = max(1, min(n - 1, forecasting.MAX_HORIZON - 1, round(n * holdout_frac)))
            context_len = max(1, n - holdout_len)

            progress.progress(i / len(sample_ids), text=f"{i + 1}/{len(sample_ids)}: {spec.name}")
            tfm_res = cached_tfm_forecast(model, sid, holdout_len, context_len)
            base_res = cached_baselines(sid, holdout_len, context_len, tuple(model_choices)) if model_choices else {}

            _spec, train_df, holdout_df = _train_holdout(sid, holdout_len, context_len)
            train_y = train_df["y"].to_numpy()
            holdout_y = holdout_df["y"].to_numpy()
            season_length = baselines.DEFAULT_SEASON_LENGTH.get(spec.freq, 1)
            all_results = {"TimesFM": tfm_res, **base_res}
            for model_name, res in all_results.items():
                if res.get("error"):
                    continue
                point = res["point"]
                q = res["quantiles"]
                # Scale-free metrics only -- MAE/RMSE/MAPE/sMAPE are deliberately excluded
                # here (see the explanation above): they aren't meaningful once averaged
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
                rows.append(row)
        progress.empty()

        if not rows:
            st.warning("No results -- every forecast in the sample errored.")
            return

        results_df = pd.DataFrame(rows)
        metric_cols = [c for c in results_df.columns if c not in ("category", "model", "n_holdout")]
        col_config = {
            c: st.column_config.NumberColumn(c, help=metrics.METRIC_INFO[c], format="%.2f")
            for c in metric_cols
            if c in metrics.METRIC_INFO
        }

        st.markdown("### By category")
        by_cat = _aggregate_summary(results_df, ["category", "model"], metric_cols)
        st.dataframe(by_cat, use_container_width=True, hide_index=True, column_config=col_config)

        st.markdown("### Overall (all sampled series)")
        overall = _aggregate_summary(results_df, ["model"], metric_cols)
        st.dataframe(overall, use_container_width=True, hide_index=True, column_config=col_config)

        chart_metric = st.selectbox("Metric to chart by category", metric_cols, key="sum_chart_metric")
        st.markdown(f"### {chart_metric} by category")
        ref_line = 1.0 if chart_metric in ("MASE", "RMSSE") else None
        if ref_line is not None:
            st.caption("Dashed line at 1.0 = seasonal-naive benchmark. Below it beats naive; above it loses to naive.")
        st.plotly_chart(plotting.summary_bar_chart(by_cat, chart_metric, reference_line=ref_line), use_container_width=True)

        st.caption(
            f"Sampled {len(sample_ids)} of {len(catalog)} total series ({per_category} per category), "
            f"holdout = {holdout_frac * 100:.0f}% of each series' own length."
        )


# ---------------------------------------------------------------------------
# Tab 5: Upload Your Own Data
# ---------------------------------------------------------------------------

FREQ_BY_LABEL = {v: k for k, v in synthetic.FREQ_LABELS.items()}


def _guess_column(cols: list[str], keywords: list[str], fallback: str) -> str:
    for kw in keywords:
        for c in cols:
            if kw in c.lower():
                return c
    return fallback


MAX_UPLOAD_ROWS = 20_000
MAX_UPLOAD_COLS = 30
MAX_COVARIATES = 10


def tab_changelog():
    changelog_path = Path(__file__).parent / "CHANGELOG.md"
    try:
        st.markdown(changelog_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        st.info("No CHANGELOG.md found next to app.py.")


def tab_upload(model):
    st.caption(
        "Upload a CSV with a date column and a numeric value column to run it through the same "
        "pipeline used for the synthetic series -- TimesFM (point + quantile) vs classical baselines, "
        "on a holdout carved out of your own data. Optionally add numeric covariate columns to try "
        "TimesFM's XReg support on your own data too. **This tool forecasts a single time series only** "
        "(one date column + one value column, optionally with a few covariate columns) -- not a panel "
        f"of multiple series. Limits: up to {MAX_UPLOAD_ROWS:,} rows, {MAX_UPLOAD_COLS} columns, and "
        f"{MAX_COVARIATES} covariates."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"], key="upload_file")
    if uploaded is None:
        st.info("Upload a CSV to get started -- it needs at least one date/time column, one numeric "
                 "value column, and 10+ rows.")
        return

    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read that file as CSV: {e}")
        return

    if raw_df.shape[1] < 2:
        st.error("Need at least two columns: a date column and a numeric value column.")
        return
    if raw_df.shape[0] > MAX_UPLOAD_ROWS:
        st.error(
            f"This file has {raw_df.shape[0]:,} rows, above this app's {MAX_UPLOAD_ROWS:,}-row limit "
            "for a single time series. Please upload a smaller file (e.g. aggregate to a coarser "
            "frequency, or trim to a shorter history)."
        )
        return
    if raw_df.shape[1] > MAX_UPLOAD_COLS:
        st.error(
            f"This file has {raw_df.shape[1]} columns, above this app's {MAX_UPLOAD_COLS}-column limit. "
            "This tool forecasts a single series (one date column + one value column, optionally with a "
            "few covariate columns), not a wide multi-series table -- please upload a narrower file."
        )
        return

    st.markdown("**Preview (first 10 rows)**")
    st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)

    cols = list(raw_df.columns)
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(raw_df[c])]
    date_guess = _guess_column(cols, ["date", "time", "ds", "dt"], cols[0])
    value_guess = _guess_column(numeric_cols or cols, ["y", "value", "sales", "target", "demand"], (numeric_cols or cols)[-1])

    left, right = st.columns([1, 3], gap="large")
    with left, st.container(border=True):
        date_col = st.selectbox("Date column", cols, index=cols.index(date_guess), key="upload_date_col")
        value_col = st.selectbox(
            "Value column to forecast", cols, index=cols.index(value_guess), key="upload_value_col"
        )
        covariate_candidates = [c for c in numeric_cols if c not in (date_col, value_col)]
        covariate_cols = st.multiselect(
            "Covariate columns (optional)", covariate_candidates, key="upload_covariate_cols",
            max_selections=MAX_COVARIATES,
            help="TimesFM will forecast using these as XReg covariates. Their values for the holdout "
            "period are taken from your own data (a backtest assumption: 'if we'd known these values "
            f"in advance'), the same way the built-in retail series are evaluated. Up to {MAX_COVARIATES}.",
        )
        freq_label = st.selectbox(
            "Data frequency", list(synthetic.FREQ_LABELS.values()), key="upload_freq",
            help="Used to pick a sensible seasonal period for the classical baselines and to label the holdout unit.",
        )
        freq = FREQ_BY_LABEL[freq_label]

    try:
        df = raw_df[[date_col, value_col, *covariate_cols]].copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    except Exception as e:
        st.error(f"Could not parse '{date_col}' as dates: {e}")
        return

    df = df.sort_values(date_col).drop_duplicates(subset=[date_col]).reset_index(drop=True)
    df = df.rename(columns={date_col: "ds", value_col: "y"})
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    n_missing_y = int(df["y"].isna().sum())
    if n_missing_y:
        df["y"] = df["y"].interpolate(limit_direction="both")
    for c in covariate_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").interpolate(limit_direction="both")

    n = len(df)
    if n < 10:
        st.error(f"Only {n} usable rows after parsing -- need at least 10 to forecast.")
        return
    if n_missing_y:
        st.warning(f"{n_missing_y} missing/non-numeric values in '{value_col}' were filled via interpolation.")

    with left, st.container(border=True):
        holdout_len, context_len = holdout_context_controls("upload", n, freq)
        model_choices = baseline_model_selector("upload", default=["SeasonalNaive", "AutoETS"])

    with right:
        train_df = df.iloc[: n - holdout_len]
        if context_len < len(train_df):
            train_df = train_df.iloc[-context_len:]
        holdout_df = df.iloc[n - holdout_len :]
        train_y = train_df["y"].to_numpy()
        holdout_y = holdout_df["y"].to_numpy()
        season_length = baselines.DEFAULT_SEASON_LENGTH.get(freq, 1)

        tfm_res = cached_tfm_forecast_upload(model, train_y, holdout_len)
        base_res = (
            cached_baselines_upload(train_df[["ds", "y"]], holdout_len, tuple(model_choices), freq)
            if model_choices
            else {}
        )

        title = f"Uploaded: {value_col} -- {holdout_len} {synthetic.FREQ_UNIT[freq]} holdout"

        if covariate_cols:
            window_df = df.iloc[: len(train_df) + holdout_len]
            dyn_num = {c: window_df[c].to_numpy() for c in covariate_cols}
            cov_res = cached_tfm_covariate_forecast_upload(model, train_y, holdout_len, dyn_num)

            st.markdown(f"**Covariates**: `{'`, `'.join(covariate_cols)}` -- plotted below the sales forecast.")
            cov_kwargs = dict(
                covariate_df=window_df, covariate_cols=covariate_cols,
                context_dates=train_df["ds"], context_y=train_y, forecast_dates=holdout_df["ds"],
                tfm_result=cov_res, holdout_y=holdout_y, baseline_results={"TimesFM (no covariates)": tfm_res, **base_res},
            )
            cov_visible = series_visibility_selector("upload", ["TimesFM (no covariates)", *model_choices])

            upload_detail_fig = plotting.covariate_fan_chart(title="Detail", zoom=True, visible=cov_visible, **cov_kwargs)
            upload_detail_fig.update_layout(showlegend=False, margin=_COMPACT_TOP_MARGIN)
            upload_overview_fig = plotting.covariate_fan_chart(title=title, zoom=False, visible=cov_visible, **cov_kwargs)
            upload_overview_fig.update_layout(showlegend=False, margin=_COMPACT_TOP_MARGIN)
            _add_zoom_rect(upload_overview_fig, upload_detail_fig, row=1, col=1)

            col1, col2 = st.columns([0.5, 0.5])
            with col1:
                st.plotly_chart(upload_overview_fig, use_container_width=True)
            with col2:
                st.plotly_chart(upload_detail_fig, use_container_width=True)

            all_results = {"TimesFM + covariates": cov_res, "TimesFM (no covariates)": tfm_res, **base_res}
        else:
            render_fan_chart_pair(
                title=title, key_prefix="upload", baseline_names=list(base_res.keys()),
                season_lengths={name: res.get("season_length") for name, res in base_res.items()},
                context_dates=train_df["ds"], context_y=train_y, forecast_dates=holdout_df["ds"],
                tfm_result=tfm_res, holdout_y=holdout_y, baseline_results=base_res,
            )
            all_results = {"TimesFM": tfm_res, **base_res}

        st.markdown("**Accuracy on holdout**")
        render_metrics_table(holdout_y, train_y, season_length, all_results)
        render_summary_stats(df["y"].to_numpy(), label=f"{value_col}, full uploaded series")
        render_calibration_section(holdout_y, tfm_res)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


APP_VERSION = "v0.1.44"


def main():
    st.title("TimesFM 2.5 Forecasting POC")
    st.caption(
        f"{APP_VERSION} -- Google TimesFM 2.5 (200M, zero-shot) vs Nixtla statsforecast baselines, "
        "on synthetic series with known ground truth."
    )
    model = get_model()
    t1, t2, t3, t4, t5, t6 = st.tabs(
        [
            "Simple Time-Series", "Covariates & Elasticity", "Hard-to-Forecast Gallery", "Summary",
            "Upload Your Own Data", "Changelog",
        ]
    )
    with t1:
        tab_fit_forecast(model)
    with t2:
        tab_covariates(model)
    with t3:
        tab_gallery(model)
    with t4:
        tab_summary(model)
    with t5:
        tab_upload(model)
    with t6:
        tab_changelog()


if __name__ == "__main__":
    main()
