# Changelog

All notable changes to this project are logged here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/). The current version is shown
in the app's subtitle (`APP_VERSION` in `app.py`) -- bump it here and there
together.

**Versioning scheme:** `v0.1.N`, where N is a running count of distinct
change requests fulfilled (a single request bundling several asks still
counts as one bump). Not standard semver-by-feature -- just an iteration
counter for a POC that hasn't cut a real release yet.

**Time estimates:** each entry below has a `Time:` line -- real wall-clock
time pulled from session message timestamps, from when the request was made
to when the change was verified, with idle gaps longer than 10 minutes
(i.e. you stepped away) excluded from the total. This is active working
time, not a productivity metric -- it's not adjusted for how many
unrelated things were happening in parallel. There is deliberately no
token-usage figure: no tool available in this session reports real
per-request token counts, and a made-up number would be worse than none.

## v0.1.32 -- 2026-08-02

Baseline snapshot of everything built so far, i.e. the first 32 change
requests. Entries below are reconstructed from the project history since
this changelog didn't exist until this point.

**Time:** ~3h45m of active work, spread across 3 separate sittings over
~1.9 calendar days (long idle gaps excluded -- see methodology above).
Not broken out per-request since most of this predates the changelog and
several early requests were only preserved as a narrative summary, not
individually timestamped.

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

## v0.1.33 -- 2026-08-02

**Time:** ~3 min

### Changed
- Config panel (left column) and results/charts (right column) now have a
  larger gap between them and the config panel is wrapped in a bordered
  container, so the two sections read as visually distinct.

## v0.1.34 -- 2026-08-02

**Time:** ~2 min

### Changed
- "Actual (holdout)" line lightened from near-black to a mid-gray
  (`#67665f`, midway between the old actual color and the history color)
  and switched to a dashed line, since the solid near-black line was
  visually dominating everything else on the chart.

## v0.1.35 -- 2026-08-02

**Time:** ~3 min (covers this and the two follow-on tweaks below, folded
into the same version at the user's request rather than bumping each time)

### Changed
- "Actual (holdout)" line iterated from dashed -> dotted -> a thin
  (width 1.0) solid line, landing on thin solid as the final look.
- Overview:detail chart width ratio changed back from 60:40 to 50:50.

## v0.1.36 -- 2026-08-02

**Time:** ~15 min (this entry itself -- required digging through session
timestamps to produce honest numbers rather than invented ones; ran
concurrently with v0.1.37 and the season-length investigation below, so
this isn't a clean, isolated duration)

### Added
- Real wall-clock time estimates on every changelog entry (see the
  methodology note near the top of this file). No token-usage figures --
  there's no tool in this session that reports real per-request token
  counts, and a fabricated number would be worse than omitting it.

## v0.1.37 -- 2026-08-02

**Time:** ~3 min

### Changed
- Series visibility selector swatches now show line style, not just
  color: baseline models render as a dotted swatch (matching their actual
  dash="dot" chart line), while History/TimesFM/Actual render as solid
  bars (matching their solid chart lines).

## v0.1.38 -- 2026-08-02

**Time:** ~15 min, mostly spent benchmarking rather than coding (see below)

### Fixed
- SeasonalNaive, Theta, and LinearRegression's Fourier term were using a
  fixed weekly (7-day) period for ALL daily-frequency series, but most of
  this catalog's daily series (Standard, Overlapping cycles, Level shift,
  Volatility regime change, retail covariates) are built with a
  LARGER-amplitude annual component than weekly, or no weekly component
  at all -- so these three models were tuned to the weaker (or wrong)
  cycle. They now use the annual period for daily data instead
  (`LONG_SEASON_LENGTH`), since it's free to compute for all three
  (benchmarked: SeasonalNaive and Theta fit in ~0.00s/~0.04s regardless of
  period). Verified directly: on an "Overlapping cycles" daily series,
  MASE dropped from 0.87 -> 0.55 (SeasonalNaive) and 0.90 -> 0.54 (Theta).
  Falls back to the weekly period on series with under ~2 years of daily
  history, so short series don't silently break (nixtla's SeasonalNaive
  fills with NaN rather than erroring when history is shorter than the
  requested period -- caught via a live warning during verification).
- AutoARIMA and AutoETS deliberately keep the weekly period for daily
  data: benchmarked directly, AutoARIMA at a 365-day period didn't finish
  a single fit in over 6 minutes (seasonal ARIMA's order search doesn't
  scale to large periods), and AutoETS was ~40x slower (18.5s vs 0.45s
  per fit) -- both too slow for an app that re-forecasts on every widget
  change. This is now a documented tradeoff (see their tooltips) rather
  than a silent blind spot, and is exactly why MSTL -- the one baseline
  that fits multiple periods at once -- tends to win on daily series with
  overlapping cycles.

## v0.1.39 -- 2026-08-02

**Time:** ~8 min

### Fixed
- The v0.1.38 season-length fix wasn't visible in the running app: the
  live Streamlit server process had `src/baselines.py` loaded in memory
  from before the fix, and Python doesn't re-import an already-loaded
  module just because its file changed on disk -- only restarting the
  server process picks up changes to imported modules (`app.py` itself
  hot-reloads on save; its imports don't). Restarted the server.

### Added
- Baseline checkbox/legend labels now show the actual lag each model
  fit with, e.g. "SeasonalNaive (lag 365)" vs "AutoETS (lag 7)", instead
  of the lag being invisible outside a tooltip -- exactly the kind of
  silent state that made the v0.1.38 bug hard to spot from the UI alone.
  MSTL, which fits multiple periods at once, shows all of them (e.g.
  "MSTL (lag 7,30,365)"). Models with no seasonal term (LinearTrend,
  ExponentialTrend) are unaffected.

## v0.1.40 -- 2026-08-02

**Time:** ~9 min

### Changed
- Replaced the v0.1.38 per-model season-length split (long period for
  SeasonalNaive/Theta, short for AutoARIMA/AutoETS, purely for cost
  reasons) with a single, principled choice shared by every single-season
  model: seasonal-strength testing (Hyndman & Athanasopoulos's STL-based
  variance-ratio measure, the same idea behind R's `findfrequency()`).
  For each frequency's candidate periods, an STL decomposition measures
  how much variance the seasonal component explains; whichever period
  scores highest (and clears a minimum strength of 0.3) is used by
  SeasonalNaive, AutoETS, Theta, and LinearRegression's Fourier term
  alike, instead of a hardcoded or cost-motivated guess. Falls back to
  season_length=1 (non-seasonal) when nothing clears the threshold.
  Verified across categories: Standard/Overlapping cycles correctly
  detect the dominant annual cycle (365), Level shift and pure
  growth/decay series correctly fall back to non-seasonal, and
  Intermittent demand -- which has no designed periodicity -- is
  explicitly guarded (series that are >50% zero skip detection outright)
  after verification showed STL spuriously reporting ~0.96 seasonal
  strength there, an overfit to sparse spikes across only ~3 annual
  cycles rather than real seasonality.
- Adds ~2s per series (STL fit at up to 3 candidate periods), covered by
  the existing `st.cache_data` layer so repeat views of the same
  series/holdout/model combination don't pay it twice.

### Removed
- AutoARIMA dropped entirely as a baseline option: with the season length
  now chosen by data rather than hardcoded, there's no fixed "safe" short
  period left to fall back to for it, and its order search was already
  benchmarked (see v0.1.38) at over 6 minutes for a single fit at a
  365-day period -- fundamentally incompatible with an app that
  re-forecasts on every widget change.
