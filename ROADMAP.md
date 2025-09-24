# Roadmap

## 0–2 weeks (useful & reproducible)

* **CLI polish**: `arkan fit|grid|eval --dataset a10 --basis bspline --p 20 --epochs 200 --out runs/...` (save JSON/CSV of metrics + best params).
* **Metrics**: add MAE, RMSE, MAPE/sMAPE, MASE; set a default metric for model selection.
* **Multi-step forecasting**: recursive vs. direct strategies (`--horizon H`).
* **Dataset plumbing**: Rdatasets local cache + dataset registry; add classics (AirPassengers, CO2, Sunspots).
* **Reproducibility**: global seed util; deterministic flags; print environment summary.
* **Examples**: 2 Colab notebooks (synthetic demo; Rdatasets grid-search).

## 2–4 weeks (model breadth & speed)

* **Exogenous inputs (ARX-KAN)** and **multivariate** (stacked AR memories).
* **Probabilistic forecasts**: quantile loss (pinball), optional conformal intervals.
* **Performance**: precompute B-spline basis; add `torch.compile` switch; microbenchmarks.
* **Baselines**: ETS/ExponentialSmoothing, SARIMA, Prophet (optional), N-BEATS (lite) for context.

## 4–8 weeks (docs, CI/CD, release)

* **Docs site** (Sphinx or mkdocs) with API reference + tutorials; publish on GitHub Pages.
* **CI**: matrix {3.10, 3.11, 3.12} + coverage badge (Codecov).
* **Packaging**: `poetry build`; optional PyPI publish on tag via GH Action; extras: `[docs]`, `[gpu]`, `[tracking]`.
* **Tracking (optional)**: MLflow/W\&B hooks behind a flag.

## Nice quality-of-life

* **Makefile** targets: `fmt`, `lint`, `type`, `test`, `bench`, `docs`.
* **Logging**: structured logs (rich/loguru) with `--log-level`.
* **Configs**: support YAML config files for experiments (`--config config.yaml`).

## Ready-to-create issues (copy/paste)

* feat(cli): add `arkan fit|grid|eval` with JSON/CSV outputs (acceptance: runs saved with metrics + params).
* feat(metrics): implement MAE, RMSE, sMAPE, MASE; default = sMAPE (acceptance: unit tests + CLI flag).
* feat(forecast): support `--horizon H` (recursive & direct) (acceptance: tests comparing strategies).
* feat(data): rdatasets local cache + registry (acceptance: offline run of examples).
* perf(bspline): precompute basis on grid + index with `torch.searchsorted` (acceptance: ≥30% faster forward).
* docs: two tutorial notebooks + Docs site with API reference.
* ci: add coverage & badge; fail under coverage < X%.
* release: GH Action to publish to PyPI on tag `v*`.
