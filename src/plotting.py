"""Plotly chart builders shared across the Streamlit app.

Color usage follows a fixed categorical order (never reassigned per filter):
Actual = near-black, TimesFM = blue, baselines = orange/aqua/yellow/magenta/green
in a fixed order. Uncertainty is shown as nested single-hue (blue) bands, light
outward from the median -- never a second color. No dual axes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#898781"
GRIDLINE = "#e1e0d9"

COLOR_ACTUAL = "#0b0b0b"
COLOR_CONTEXT = "#c3c2b7"
COLOR_TIMESFM = "#2a78d6"
# Ordered so the two lower-contrast hues in this palette (magenta, aqua -- both
# measured below 3:1 contrast on a light surface) land on Theta/MSTL, the two
# baselines that are opt-in rather than selected by default, while the
# default-shown/most-common baselines (SeasonalNaive, AutoETS, AutoARIMA) get the
# higher-contrast hues (orange, green, violet).
BASELINE_COLORS = {
    "SeasonalNaive": "#eb6834",  # orange
    "AutoETS": "#008300",  # green
    "AutoARIMA": "#4a3aa7",  # violet
    "Theta": "#e87ba4",  # magenta (lower contrast -- opt-in model)
    "MSTL": "#1baf7a",  # aqua (lower contrast -- opt-in model)
}
BAND_FILLS = {
    (0.1, 0.9): "rgba(42,120,214,0.12)",
    (0.2, 0.8): "rgba(42,120,214,0.18)",
    (0.3, 0.7): "rgba(42,120,214,0.26)",
    (0.4, 0.6): "rgba(42,120,214,0.34)",
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK_PRIMARY, size=13),
    # Charts often render in half-width columns (the overview/detail pair sits side by
    # side), which wraps a 4-5-item horizontal legend onto two lines -- font/tracegap
    # are tightened to make that less likely, and the top margin is sized generously
    # enough to hold the title above a *wrapped* two-line legend, not just one line.
    legend=dict(
        orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
        font=dict(size=11), tracegroupgap=4, itemwidth=30,
    ),
    margin=dict(l=10, r=10, t=150, b=10),
    hovermode="x unified",
)
AXIS_DEFAULTS = dict(gridcolor=GRIDLINE, zeroline=False, linecolor=GRIDLINE)
YAXIS_DEFAULTS = dict(**AXIS_DEFAULTS, rangemode="tozero")
EVENT_COLOR = "#e34948"  # red -- distinct from all series colors, reserved for structural-break markers


def _title(text: str) -> dict:
    """Left-aligned title, letting Plotly auto-place it vertically within the
    (now generous) top margin rather than hand-picking a y that could clip
    off-canvas."""
    return dict(text=text, x=0, xanchor="left")


def _add_fan_traces(
    fig: go.Figure,
    context_dates: pd.Series,
    context_y: np.ndarray,
    forecast_dates: pd.Series,
    tfm_result: dict,
    holdout_y: np.ndarray | None,
    baseline_results: dict[str, dict] | None,
    event_dates: list | None,
    zoom: bool,
    context_window: int | None,
    row: int | None = None,
    col: int | None = None,
    visible: set[str] | None = None,
) -> tuple[pd.Series, np.ndarray]:
    """Adds history/quantile-band/point/actual/baseline/event traces to `fig`
    (optionally at a specific subplot row/col). Returns the (possibly
    zoom-trimmed) history dates/values, since callers need them afterward to
    compute a tight zoom axis range.

    visible: if given, a set of series-name labels to actually draw (the same
    labels used for legend names -- "History", "TimesFM (point)", "TimesFM
    quantile bands", "Actual (holdout)", plus each baseline's name). None means
    show everything. Hidden series are omitted entirely (not just dimmed), so
    they never affect the axis range either.
    """
    show_all = visible is None

    plot_dates = context_dates
    plot_y = context_y
    if zoom:
        lead_in = max(3, len(forecast_dates) // 3)
        if len(plot_dates) > lead_in:
            plot_dates = plot_dates.iloc[-lead_in:]
            plot_y = plot_y[-lead_in:]
    elif context_window is not None and len(plot_dates) > context_window:
        plot_dates = plot_dates.iloc[-context_window:]
        plot_y = plot_y[-context_window:]

    if show_all or "History" in visible:
        fig.add_trace(
            go.Scatter(
                x=plot_dates, y=plot_y, name="History", mode="lines",
                line=dict(color=COLOR_CONTEXT, width=1.0), hoverinfo="skip",
            ),
            row=row, col=col,
        )

    quantiles = tfm_result["quantiles"]
    if show_all or "TimesFM quantile bands" in visible:
        for (qlo, qhi), fill in BAND_FILLS.items():
            fig.add_trace(
                go.Scatter(
                    x=list(forecast_dates) + list(forecast_dates[::-1]),
                    y=list(quantiles[qhi]) + list(quantiles[qlo][::-1]),
                    fill="toself", fillcolor=fill, line=dict(width=0), mode="lines",
                    name=f"TimesFM {int(qlo*100)}-{int(qhi*100)}%", hoverinfo="skip", showlegend=False,
                ),
                row=row, col=col,
            )

    if show_all or "TimesFM (point)" in visible:
        fig.add_trace(
            go.Scatter(
                x=forecast_dates, y=tfm_result["point"], name="TimesFM (point)", mode="lines",
                line=dict(color=COLOR_TIMESFM, width=1.5),
            ),
            row=row, col=col,
        )

    if baseline_results:
        for name, res in baseline_results.items():
            if res.get("error") or not (show_all or name in visible):
                continue
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates, y=res["point"], name=name, mode="lines",
                    line=dict(color=BASELINE_COLORS.get(name, INK_SECONDARY), width=1.25, dash="dot"),
                ),
                row=row, col=col,
            )

    if holdout_y is not None and (show_all or "Actual (holdout)" in visible):
        fig.add_trace(
            go.Scatter(
                x=forecast_dates, y=holdout_y, name="Actual (holdout)", mode="lines",
                line=dict(color=COLOR_ACTUAL, width=1.5),
            ),
            row=row, col=col,
        )

    # The split marker and event markers deliberately do NOT pass row/col: xref="x" +
    # yref="paper" (Plotly's convention, used by add_vline internally too) anchors the
    # x-position to the primary/first subplot's axis but spans the *entire* figure height
    # top to bottom -- which for a single-panel fan_chart is just that one panel, and for
    # the combined covariate_fan_chart draws one line through the main panel *and* every
    # covariate row below it (since shared_xaxes keeps them all on the same x-scale).
    if len(context_dates) > 0:
        split_x = context_dates.iloc[-1]
        fig.add_vline(x=split_x, line_width=1, line_dash="dash", line_color=INK_SECONDARY)

    for i, ev in enumerate(event_dates or []):
        # add_vline(..., annotation_text=...) is used instead of add_shape+add_annotation
        # elsewhere in this file, but it's broken for a datetime x-axis + annotation combo
        # in this plotly/pandas pairing (its internal midpoint calc does sum([x0, x1])
        # starting from int 0, which pandas Timestamps reject). Add the shape and the
        # annotation as two separate calls instead.
        fig.add_shape(
            type="line", x0=ev, x1=ev, y0=0, y1=1, xref="x", yref="paper",
            line=dict(width=2, dash="dot", color=EVENT_COLOR),
        )
        if i == 0:
            fig.add_annotation(
                x=ev, y=1, xref="x", yref="paper", text="event", showarrow=False,
                yanchor="bottom", font=dict(color=EVENT_COLOR, size=11),
            )

    return plot_dates, plot_y


def _zoom_y_range(values: list[float]) -> list[float]:
    lo, hi = float(np.min(values)), float(np.max(values))
    pad = (hi - lo) * 0.08 or max(abs(hi), 1.0) * 0.08
    return [lo - pad, hi + pad]


def fan_chart(
    context_dates: pd.Series,
    context_y: np.ndarray,
    forecast_dates: pd.Series,
    tfm_result: dict,
    holdout_y: np.ndarray | None = None,
    baseline_results: dict[str, dict] | None = None,
    title: str = "",
    context_window: int | None = None,
    event_dates: list | None = None,
    zoom: bool = False,
    visible: set[str] | None = None,
) -> go.Figure:
    """Historical context + TimesFM quantile fan + point forecast, actual
    holdout overlay, and optional baseline model lines.

    context_window: if set, only the last N context points are shown (e.g. to
    zoom into recent history on very long series). Defaults to showing the
    full context, since truncating can silently crop out the very event
    (level shift, volatility break) the chart is meant to show.
    event_dates: dates of known structural events (level shifts, volatility
    breaks) to mark with a labeled vertical line, so they're never missed
    regardless of context length or noise.
    zoom: if True, produce a "detail" view -- x-axis tightly framed to a short
    lead-in plus the forecast window, and y-axis tightly framed to the actual
    visible values (not zero-anchored), for reading off the forecast precisely.
    context_window is ignored when zoom=True (zoom sets its own lead-in).
    visible: optional set of series-name labels to draw; None shows everything.
    Hidden series are left out of the zoom axis range too, not just undrawn.
    """
    show_all = visible is None
    fig = go.Figure()
    plot_dates, plot_y = _add_fan_traces(
        fig, context_dates, context_y, forecast_dates, tfm_result, holdout_y, baseline_results,
        event_dates, zoom, context_window, visible=visible,
    )
    fig.update_layout(title=_title(title), **LAYOUT_DEFAULTS)

    if zoom:
        quantiles = tfm_result["quantiles"]
        visible_y = []
        if show_all or "History" in visible:
            visible_y += list(plot_y)
        if show_all or "TimesFM quantile bands" in visible:
            visible_y += list(quantiles[0.1]) + list(quantiles[0.9])
        if show_all or "TimesFM (point)" in visible:
            visible_y += list(tfm_result["point"])
        if holdout_y is not None and (show_all or "Actual (holdout)" in visible):
            visible_y += list(holdout_y)
        for name, res in (baseline_results or {}).items():
            if not res.get("error") and (show_all or name in visible):
                visible_y += list(res["point"])
        if not visible_y:  # everything hidden -- fall back to the point forecast so the axes stay sane
            visible_y = list(tfm_result["point"])
        x_start = plot_dates.iloc[0] if len(plot_dates) > 0 else forecast_dates.iloc[0]
        fig.update_xaxes(range=[x_start, forecast_dates.iloc[-1]], **AXIS_DEFAULTS)
        fig.update_yaxes(range=_zoom_y_range(visible_y), **AXIS_DEFAULTS)
    else:
        fig.update_xaxes(**AXIS_DEFAULTS)
        fig.update_yaxes(**YAXIS_DEFAULTS)
    return fig


def covariate_fan_chart(
    covariate_df: pd.DataFrame,
    covariate_cols: list[str],
    context_dates: pd.Series,
    context_y: np.ndarray,
    forecast_dates: pd.Series,
    tfm_result: dict,
    holdout_y: np.ndarray | None = None,
    baseline_results: dict[str, dict] | None = None,
    title: str = "",
    zoom: bool = False,
    visible: set[str] | None = None,
) -> go.Figure:
    """Main sales history+forecast panel on top, one covariate small-multiple
    directly underneath it per covariate, all sharing the same x-axis -- so you
    can read off exactly what price/promo/holiday were doing at any point the
    forecast reacts to, without cross-referencing a separate chart.

    visible: optional set of series-name labels to draw in the top panel; None
    shows everything (same convention as fan_chart's `visible`).
    """
    show_all = visible is None
    n_cov = len(covariate_cols)
    row_heights = [0.55] + [0.45 / n_cov] * n_cov
    fig = make_subplots(
        rows=1 + n_cov, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=row_heights,
        subplot_titles=["Sales -- history & forecast"] + [c.replace("_", " ").title() for c in covariate_cols],
    )

    plot_dates, plot_y = _add_fan_traces(
        fig, context_dates, context_y, forecast_dates, tfm_result, holdout_y, baseline_results,
        None, zoom, None, row=1, col=1, visible=visible,
    )

    palette = list(BASELINE_COLORS.values()) + [COLOR_TIMESFM]
    split_date = context_dates.iloc[-1] if len(context_dates) > 0 else None
    x_start = plot_dates.iloc[0] if len(plot_dates) > 0 else forecast_dates.iloc[0]
    x_end = forecast_dates.iloc[-1]

    for i, col in enumerate(covariate_cols):
        r = i + 2
        values = covariate_df[col].to_numpy()
        is_flag = set(np.unique(values)).issubset({0.0, 1.0})  # 0/1 indicators (promo, holiday) vs. e.g. price
        fig.add_trace(
            go.Scatter(
                x=covariate_df["ds"], y=values, name=col, mode="lines",
                line=dict(color=palette[i % len(palette)], width=1.25), showlegend=False,
            ),
            row=r, col=1,
        )
        if split_date is not None:
            fig.add_vline(x=split_date, line_width=1, line_dash="dash", line_color=INK_SECONDARY, row=r, col=1)
        if zoom:
            window = covariate_df[(covariate_df["ds"] >= x_start) & (covariate_df["ds"] <= x_end)][col]
            fig.update_yaxes(range=_zoom_y_range(list(window)), row=r, col=1, **AXIS_DEFAULTS)
        elif is_flag:
            fig.update_yaxes(row=r, col=1, **YAXIS_DEFAULTS)
        else:
            # Non-flag covariates (e.g. price) float freely -- forcing a narrow $8-12
            # band to include 0 would flatten out all the visible variation.
            fig.update_yaxes(row=r, col=1, **AXIS_DEFAULTS)
        fig.update_xaxes(row=r, col=1, **AXIS_DEFAULTS)

    fig.update_layout(title=_title(title), **LAYOUT_DEFAULTS, height=420 + 130 * n_cov)
    for ann in fig["layout"]["annotations"][: 1 + n_cov]:
        ann["font"] = dict(size=12, color=INK_SECONDARY)

    if zoom:
        quantiles = tfm_result["quantiles"]
        visible_y = []
        if show_all or "History" in visible:
            visible_y += list(plot_y)
        if show_all or "TimesFM quantile bands" in visible:
            visible_y += list(quantiles[0.1]) + list(quantiles[0.9])
        if show_all or "TimesFM (point)" in visible:
            visible_y += list(tfm_result["point"])
        if holdout_y is not None and (show_all or "Actual (holdout)" in visible):
            visible_y += list(holdout_y)
        for name, res in (baseline_results or {}).items():
            if not res.get("error") and (show_all or name in visible):
                visible_y += list(res["point"])
        if not visible_y:
            visible_y = list(tfm_result["point"])
        fig.update_xaxes(range=[x_start, x_end])
        fig.update_yaxes(range=_zoom_y_range(visible_y), row=1, col=1, **AXIS_DEFAULTS)
    else:
        fig.update_xaxes(row=1, col=1, **AXIS_DEFAULTS)
        fig.update_yaxes(row=1, col=1, **YAXIS_DEFAULTS)
    return fig


def elasticity_chart(forecast_dates: pd.Series, base_point: np.ndarray, bumped_point: np.ndarray, pct_bump: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=base_point,
            name="Base price forecast",
            mode="lines",
            line=dict(color=COLOR_TIMESFM, width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=bumped_point,
            name=f"Price +{pct_bump*100:.0f}% forecast",
            mode="lines",
            line=dict(color=BASELINE_COLORS["SeasonalNaive"], width=1.5, dash="dot"),
        )
    )
    fig.update_layout(title=_title("Demand response to a price change"), **LAYOUT_DEFAULTS)
    fig.update_xaxes(**AXIS_DEFAULTS)
    fig.update_yaxes(**YAXIS_DEFAULTS)
    return fig


def calibration_chart(calib_df: pd.DataFrame) -> go.Figure:
    """Reliability diagram: nominal quantile level vs empirical coverage.
    Fixed [0,1]-[0,1] axes (not zero-anchored-and-autoscaled like value charts)
    since this is a diagnostic against the y=x diagonal, not a magnitude plot."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
            line=dict(color=INK_SECONDARY, width=1.0, dash="dash"), hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=calib_df["nominal quantile"], y=calib_df["empirical coverage"], mode="lines+markers",
            name="TimesFM", line=dict(color=COLOR_TIMESFM, width=1.5), marker=dict(size=6),
        )
    )
    fig.update_layout(title=_title("Quantile calibration: nominal vs empirical"), **LAYOUT_DEFAULTS)
    fig.update_xaxes(title="Nominal quantile", range=[0, 1], **AXIS_DEFAULTS)
    fig.update_yaxes(title="Empirical coverage (fraction of actuals ≤ forecast)", range=[0, 1], **AXIS_DEFAULTS)
    return fig


def summary_bar_chart(agg_df: pd.DataFrame, metric: str, reference_line: float | None = None) -> go.Figure:
    """Grouped bar chart: one bar-group per category, one bar per model, for a
    single metric. Fixed model->color mapping (never reassigned by filtering).
    reference_line: e.g. 1.0 for MASE/RMSSE, marking the seasonal-naive benchmark."""
    color_map = {"TimesFM": COLOR_TIMESFM, **BASELINE_COLORS}
    categories = list(dict.fromkeys(agg_df["category"]))
    models = list(dict.fromkeys(agg_df["model"]))
    fig = go.Figure()
    for m in models:
        sub = agg_df[agg_df["model"] == m].set_index("category").reindex(categories)
        fig.add_trace(
            go.Bar(x=categories, y=sub[metric], name=m, marker_color=color_map.get(m, INK_SECONDARY))
        )
    if reference_line is not None:
        fig.add_hline(y=reference_line, line_width=1.5, line_dash="dash", line_color=EVENT_COLOR)
    fig.update_layout(title=_title(f"{metric} by category"), barmode="group", **LAYOUT_DEFAULTS)
    fig.update_xaxes(**AXIS_DEFAULTS)
    fig.update_yaxes(**YAXIS_DEFAULTS)
    return fig
