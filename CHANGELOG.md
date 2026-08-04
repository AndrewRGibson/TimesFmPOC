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

## v0.1.41 -- 2026-08-03

**Time:** ~20 min

### Changed
- Renamed the "Fit & Forecast" tab to "Simple Time-Series".
- Added a "Changelog" tab that renders this file directly, instead of it
  only being readable from the repo.
- Main forecast fan charts (overview/detail pairs) are 20% shorter
  (360px vs Plotly's 450px default) so more fits on the page. Scoped
  narrowly to the charts that already drop their legend/use a compact
  margin (`_COMPACT_TOP_MARGIN`) -- verified directly via a first attempt
  at a global height cut (through the shared `LAYOUT_DEFAULTS`), which
  clipped the title and y-axis label on `calibration_chart` and would
  have done the same to `elasticity_chart`/`summary_bar_chart`: those
  keep a real legend and a taller default margin that a global height
  cut doesn't leave enough room for.
- Dropped the redundant `st.subheader` title at the top of every tab
  (e.g. "Fit & Forecast", "Summary: Performance Across All Series") --
  the tab selector already names the page; each tab's descriptive
  caption is unchanged and still explains what it does.
- Summary tab's "80% coverage %" aggregation (by category and overall)
  is now weighted by each row's holdout length instead of averaged one
  series/model = one vote. Coverage is an empirical hit rate, so pooling
  by point count is the statistically standard way to combine rates from
  groups of very different sample sizes -- verified directly: a 4-point
  series at 0% coverage averaged against a 200-point series at 82%
  coverage now aggregates to ~80.4% (correctly close to the pooled true
  rate), not the naive 41% a plain mean-of-means would give, which let a
  single tiny, noisy holdout swing the aggregate as much as a large,
  reliable one. Only coverage changed -- MASE/RMSSE/scaled pinball stay
  simple per-series averages, which is still the right call for those
  (see the updated in-app explanation on the Summary tab).

## v0.1.42 -- 2026-08-03

**Time:** ~5 min

### Added
- "scaled width" metric: the 80% interval width divided by the same
  in-sample seasonal-naive scale MASE/scaled-pinball already use, making
  it comparable across series of different magnitude (plain "80% width"
  is in the series' own units, same limitation as MAE/RMSE). Answers the
  "is that just scaled pinball?" question from last revision -- no,
  scaled pinball jointly measures calibration and sharpness together;
  this isolates sharpness (band width) alone, meant to be read alongside
  "80% coverage %" rather than replace it. Shown on the per-series
  metrics tables and included in the Summary tab's sweep (simple
  per-series average there, like MASE/RMSSE -- not weighted by holdout
  length like coverage, since it's a ratio rather than a hit rate).

## v0.1.43 -- 2026-08-03

**Time:** ~24 min

### Changed
- "Standard" category no longer includes daily-frequency series (now
  weekly/monthly/annual only) -- daily data in this catalog always
  carries a real (often dominant) annual cycle on top of any weekly one
  (see the v0.1.40 seasonal-strength work), making it a multi-cycle case
  rather than the simple one "Standard"/"Simple Time-Series" is for.
  Every `HARD_CATEGORIES` category (Short series, Level shift,
  Overlapping cycles, Intermittent demand, Volatility regime change,
  Growth & decay traps) already included daily instances and keeps them
  -- that's where daily's complexity now exclusively lives. "Covariates /
  elasticity" is intentionally unchanged (still Daily+Weekly): it's tied
  to its own tab's elasticity/XReg UI which the Gallery tab doesn't
  render, and daily retail data is realistic rather than a "hard"
  edge case.
- Catalog size per category dropped from ~10 to ~5 instances (40 total
  series, down from 80). "Growth & decay traps" (built from two
  sub-generators, exponential growth + rapid decay) splits 3/2 rather
  than a flooring 2/2, so it still totals 5 rather than quietly landing
  one short.

## v0.1.44 -- 2026-08-03

**Time:** ~13 min

### Fixed
- AutoETS and Theta were both silently failing to pick up real, strong
  seasonality despite correctly detecting and displaying the right lag
  -- confirmed as two separate upstream nixtla bugs, not a false alarm,
  using the first weekly "Simple Time-Series" series (`standard_1000`,
  52-week detected lag) as the reproduction case:
  - **AutoETS**: its default `model="ZZZ"` full search chose
    `ETS(A,Ad,N)` (no season) at AIC=1167.5, while forcing an additive
    seasonal search (`model="ZZA"`) found `ETS(A,A,A)` at AIC=1155.9 --
    decisively better, but never reached by the unconstrained search.
    Fixed by passing `model="ZZA"` whenever `detect_season_length` has
    already confirmed real seasonality (season_length > 1); `season_length
    == 1` keeps the original unconstrained `"ZZZ"` search.
  - **Theta**: its built-in seasonality test only checks whether the ACF
    at exactly lag `season_length` clears a Bartlett-formula threshold --
    on `standard_1000` that scored 1.25 against a 1.645 cutoff ("not
    significant"), even though our own STL-based test found a strength
    of 0.898. A trend inflates ACF at every lag, which is exactly what
    makes this kind of single-lag, undetrended test unreliable. Fixed by
    pre-deseasonalizing with the same `statsmodels.seasonal_decompose`
    Theta uses internally whenever we've already confirmed seasonality,
    fitting plain Theta on the deseasonalized series, then adding the
    (correctly phase-aligned, repeating) seasonal component back onto
    the point forecast and every quantile.
  - Verified directly on `standard_1000`: MASE improved from what a flat,
    non-seasonal forecast would score (>1) to 0.28 (AutoETS) and 0.36
    (Theta) -- now in the same range as MSTL (0.31) and LinearRegression
    (0.27), a coherent picture across all four seasonal-aware baselines
    instead of two working and two silently broken. Re-verified on a
    second weekly series and spot-checked monthly/annual/daily frequencies
    for regressions -- none found.

## v0.1.45 -- 2026-08-03

**Time:** ~6 min

### Added
- "calculation ms" column: wall-clock time to fit + forecast, per model,
  measured around the actual fit/forecast call only (excludes one-time
  shared prep like season-length detection). Shown on every per-series
  metrics table (Simple Time-Series, Covariates, Gallery, Upload) and in
  the Summary tab's sweep (simple per-series average there, like
  MASE/RMSSE). Cached results show the original computation's time, not
  near-zero cache-retrieval time, since that's the number worth knowing.
  Verified directly: AutoETS's ~103ms on a weekly series is visibly
  higher than SeasonalNaive's ~6ms or LinearRegression's ~5ms, putting a
  real number on cost differences the app previously only asserted in
  tooltips ("relatively slow to fit").

### Changed
- Renamed "Accuracy on holdout" to "Accuracy & calculation time on
  holdout" (and the Covariates tab's equivalent) to reflect the new
  column.

## v0.1.46 -- 2026-08-03

**Time:** ~12 min

### Changed
- Summary tab now loads precomputed results from `data/summary_precomputed.json`
  instead of running a live sweep. Diagnosed first rather than guessing: per-series
  compute is genuinely fast (TimesFM ~270-800ms even at a full 931-point daily
  context, baselines a few ms to ~130ms for AutoETS -- measured directly with the
  v0.1.45 `calc_ms` instrumentation), but the sweep loop re-ran in full on every
  widget interaction anywhere in the app (Streamlit reruns the whole script on any
  interaction, and the tab's `sum_ran` gate only skipped the *initial* run, not
  every rerun after), and many fast per-series calls across dozens of series and
  models still add up. Fixed by factoring the row-computation logic into
  `src/summary_compute.py` (shared, so the live tab and the offline script can't
  drift apart) and adding `scripts/precompute_summary.py`, which runs the full
  sweep once -- all 40 catalog series x TimesFM + all 7 baselines, 316 rows in
  232s -- and writes the result to a committed JSON file the tab just reads.
  Individual-series tabs (Simple Time-Series, Covariates, Gallery, Upload) are
  unaffected -- still computed live/on demand as before, per the user's explicit
  direction to keep those interactive and only lock down the Summary rollup.
  The Summary tab's sample-size/model-choice/holdout-fraction sliders and "Run"
  button are gone along with the live sweep; re-run the script (documented in its
  own docstring) after changing the synthetic catalog or model.

## v0.1.47 -- 2026-08-04

**Time:** ~10 min

### Added
- "80% coverage SE %" column on the Summary tab's "By category" and "Overall"
  aggregate tables, next to the pooled "80% coverage %" they already showed.
  Computed as the *pooled* binomial standard error (`sqrt(p(1-p)/N)` over the
  combined holdout-point count across every series in the group), not an
  average of each series' individual SE -- the per-series SE column already
  shown elsewhere in the app would overstate the aggregate's uncertainty if
  simply averaged, since it ignores that the series are pooled. Verified
  against a hand-computed example (two series, n=10 and n=200, weighted
  coverage 75.24% -> SE 2.98%) before shipping.

## v0.1.48 -- 2026-08-04

**Time:** ~30 min

### Fixed
- Baseline model colors/line-styles, across every chart that draws them
  (fan charts, the Covariates combined chart, the Summary tab's bar chart).
  `BASELINE_COLORS` only had entries for 4 of the 7 selectable baselines --
  LinearTrend, LinearRegression, and ExponentialTrend all silently fell back
  to the same gray, making them indistinguishable from each other whenever
  more than one was checked (reported directly from a screenshot showing all
  three as identical dotted gray lines). Fixed by giving all 7 a fixed color,
  chosen by running this repo's dataviz-skill palette validator (OKLab CVD +
  normal-vision separation) over every permutation of the candidate hues
  against the 7 fixed model slots. That search also caught a pre-existing
  defect: the two *default-shown* baselines, SeasonalNaive and AutoETS, were
  orange/green -- a pairing that fails the colorblind-separation check
  (ΔE 3.2 protanopia, a classic red-green collision) despite being the first
  thing every user sees; they're now orange/aqua, which passes cleanly. Also
  added a new `BASELINE_DASHES` mapping (each baseline gets its own dash
  style, not a shared "dot") as a secondary, non-color identity channel --
  with 7 independently-toggleable series, no hue set can stay pairwise-safe
  under simulated colorblindness once more than ~3 are shown at once (the
  validator confirms this is a hard limit of the palette, not a fixable
  ordering problem), so color and dash are placed to never both collide on
  the same pair. Verified by rendering all 7 baselines together and visually
  confirming each is distinguishable.
