# Changelog

All notable changes to this project are logged here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/). The current version is shown
in the app's subtitle (`APP_VERSION` in `app.py`) -- bump it here and there
together.

**Versioning scheme:** `v0.1.N`, where N is a running count of distinct
change requests fulfilled (a single request bundling several asks still
counts as one bump). Not standard semver-by-feature -- just an iteration
counter for a POC that hasn't cut a real release yet.

## v0.1.32 -- 2026-08-02

Baseline snapshot of everything built so far, i.e. the first 32 change
requests. Entries below are reconstructed from the project history since
this changelog didn't exist until this point.

### Added
- Initial POC: TimesFM 2.5 (200M, zero-shot torch) forecasting with quantile
  fans vs Nixtla `statsforecast` baselines, on synthetic series with known
  ground truth; holdout-period UI and series switching.
- Covariates & elasticity: TimesFM XReg support, a price-elasticity
  what-if chart, and OLS-based diagnostic statistics for extracted coefficients.
- Summary tab: cross-category performance rollup, an AutoARIMA baseline,
  and a balanced synthetic catalog (8 categories x 10 series, mixed
  daily/weekly/monthly/annual frequency).
- Hard-to-forecast gallery, including a dedicated "Growth & decay traps"
  category (exponential growth and rapid-decay series).
- Paired overview/detail chart layout (60:40 width split) with a light
  zoom-region outline on the overview chart, decluttered titles/legends,
  and free-floating (non-zero-anchored) price axes.
- M4/M5-competition-style accuracy metrics (MASE, RMSSE, scaled pinball
  loss) with detailed tooltips explaining what each metric is for and its
  strengths/weaknesses, plus a standard error on 80% coverage estimates.
- Regression baseline models: LinearTrend, LinearRegression (Fourier
  seasonal terms), and ExponentialTrend (log-linear, for series with
  exponential growth/decline).
- Per-model tooltips (`MODEL_INFO`) documenting what each baseline does and
  does not account for (e.g. SeasonalNaive ignoring trend).
- Checkbox-based series visibility selector and baseline-model selector,
  color-matched exactly to the chart legend (AutoARIMA flagged as
  relatively slow); native Plotly legends removed once the selectors made
  them redundant.
- Auto-run forecasts on startup and on any control change (Summary tab
  keeps its explicit "Run" button given the cost of a full sweep).
- "Upload Your Own Data" tab: single-time-series CSV upload (optional
  covariates), with row (20,000), column (30), and covariate (10) limits.
- Version number surfaced in the app subtitle.
- MIT license, README.

### Fixed
- A reproducible Streamlit/PyTorch segfault (`torch.classes.__path__`
  introspection during Streamlit's caching/file-watching) -- fixed by
  clearing `torch.classes.__path__` immediately after import.
- A `_cycle_combos` sampling bug that collapsed several synthetic
  categories onto a single frequency instead of a balanced mix.
- A backcast-trim bug in the forecasting pipeline.
- Rapid-decay synthetic series timed so the decay event lands near the
  holdout boundary, instead of resolving to zero long before it starts.
- Stray colored dot markers on the quantile-fan bands (Plotly defaulting
  short traces to `lines+markers`); band traces now force `mode="lines"`.
