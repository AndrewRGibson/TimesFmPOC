# TimesFM Forecasting POC

Interactive Streamlit app for exploring Google's [TimesFM 2.5](https://github.com/google-research/timesfm) foundation model on synthetic time series, comparing it against classical baselines from Nixtla's `statsforecast`.

## Features

- **Fit & Forecast** -- general-purpose forecasting on "Standard" series (trend + seasonality, easy baseline cases). Pick a series, set a holdout window, see TimesFM's point + quantile fan chart forecast vs actuals alongside classical baselines, with an overview + zoomed-detail chart pair and a per-series metrics table (MASE, RMSSE, MAE, MAPE, sMAPE, RMSE, coverage, pinball loss).
- **Covariates & Elasticity** -- forecast retail-style series with price/promotion/holiday covariates via TimesFM's XReg support (combined chart: sales forecast on top, each covariate plotted directly below on a shared x-axis). Estimates price elasticity two ways: a perturbation-based estimate from TimesFM's own forecast, and an independent classical OLS regression (with standard errors, p-values, R², and a trend-confound check) -- both checked against the synthetic data's known ground-truth elasticity.
- **Hard-to-Forecast Gallery** -- compare TimesFM vs baselines across deliberately difficult scenarios: very short series, abrupt level shifts, overlapping seasonal cycles, intermittent demand, volatility regime changes (in both directions), and growth/decay traps (apparent-exponential growth that saturates; rapid decay toward the non-negativity floor).
- **Summary** -- runs TimesFM + baselines across a sample of series from every category and aggregates accuracy by category and overall, using scale-free metrics (MASE, RMSSE, scaled pinball loss -- the M4/M5 competition approach) rather than raw MAE/RMSE/MAPE, which aren't meaningful once averaged across series of very different scale.

All series are synthetic (see `src/synthetic.py`), spanning daily/weekly/monthly/annual frequency, so ground truth and holdout actuals are always available. Every chart lets you toggle which series/lines are visible, and shows an overview + zoomed-detail pair at a 2:1 width ratio.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

First run downloads the ~200M-parameter TimesFM 2.5 checkpoint from Hugging Face (~800MB).

## Run

```
.venv\Scripts\streamlit run app.py --server.fileWatcherType none
```

`--server.fileWatcherType none` avoids a known Streamlit/PyTorch incompatibility: Streamlit's
default file watcher introspects every imported module's `__path__` to support hot-reload, and
`torch.classes.__path__` raises inside that introspection in a way that has segfaulted the
process in testing. Since the watcher is only needed for live code-reload during development,
disabling it is the simplest fix; re-run `streamlit run app.py` manually after editing files.

## License

MIT -- see [LICENSE](LICENSE). Provided as-is, no warranty.
