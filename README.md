# AR-KAN Time Series

AR-KAN for time series in Python: an **AR(p) memory** learned from data, feeding a **KAN-style** nonlinear stage. This repo includes:

* A compact AR memory (Yule–Walker / Levinson–Durbin)
* KAN blocks with **RBF** or **cubic B-spline** univariate bases
* A full **ARKAN** model for one-step-ahead forecasting
* **Grid-search** over ARIMA and AR-KAN hyperparameters
* **STL Periodicity Strength** and an **Rdatasets** loader
* Examples and a minimal CLI

> Paper implemented: **Zeng, Xu, & Wang (2025)** — *AR‑KAN: Autoregressive‑Weight‑Enhanced Kolmogorov–Arnold Network for Time Series Forecasting* (arXiv:2509.02967v2, [https://arxiv.org/abs/2509.02967](https://arxiv.org/abs/2509.02967)).

---

## Maintainer

* **Diogo Ribeiro** (GitHub: [@diogoribeiro7](https://github.com/diogoribeiro7))
  Affiliation: **ESMAD**; **Mysense.ai**
  ORCID: **0009-0001-2022-7072**
  Email: **[dfr@esmad.ipp.pt](mailto:dfr@esmad.ipp.pt)**

---

## Citations

### Citing this repository

```bibtex
@misc{ribeiro2025arkan_ts,
  title        = {AR-KAN Time Series: AR Memory + KAN Nonlinearity in Python},
  author       = {Diogo Ribeiro},
  year         = {2025},
  howpublished = {GitHub repository},
  url          = {https://github.com/diogoribeiro7/ar-kan-timeseries},
  institution  = {ESMAD and Mysense.ai}
}
```

### Citing the paper we implement

```bibtex
@article{zeng2025arkan,
  title   = {AR-KAN: Autoregressive-Weight-Enhanced Kolmogorov--Arnold Network for Time Series Forecasting},
  author  = {Zeng, Chen and Xu, Tiehang and Wang, Qiao},
  journal = {arXiv preprint arXiv:2509.02967},
  year    = {2025},
  url     = {https://arxiv.org/abs/2509.02967}
}
```
