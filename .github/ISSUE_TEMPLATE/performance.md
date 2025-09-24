# Performance

Guidance for benchmarking and avoiding regressions.

## 1\. Metrics & Protocol

- **Primary metric**: Test MSE on the last 20% (80/20 split).
- **Synthetic sets**: `f1(t) = cos(2t) + cos(2πt) + ε`, `f2(t) = sin(3t) + sin(2e t) + ε`.
- **Noise σ**: {0.1, 0.2, 0.3, 0.4}. N=500, t ∈ [0, 8π].
- **Training**: Adam, lr=3e-3, epochs=200–300, batch=64, device=CPU by default.
- **Models**: AR-KAN (vary p, hidden, n_basis, basis), ARIMA baseline (grid).

## 2\. Reproducible Runs

**Synthetic (CLI)**

```bash
poetry run arkan demo --basis bspline --epochs 200
poetry run arkan grid --basis bspline --epochs 200
```

**Rdatasets (examples)**

```bash
poetry run python examples/demo_rdatasets.py
```

## 3\. Expected Ranges (sanity checks)

- On f1 with σ≈0.2, AR-KAN(bspline) typically achieves test MSE below ARIMA for reasonable `p` (e.g., 20), given enough epochs (≥200). The exact numbers depend on RNG and hardware; treat values as order-of-magnitude checks rather than fixed targets.
- For series with stronger seasonal component (higher periodicity strength), AR-KAN tends to close or surpass ARIMA.

## 4\. Complexity (rough)

- **ARMemory (fit)**: Levinson–Durbin is ~O(p²), plus autocovariances O(p·T).
- **ARMemory (transform)**: O(T·p).
- **KANBlock**: for batch B, input d, basis M, out_dim k → O(B·(d·M·k)). Two blocks dominate the forward.
- **Training**: cost scales with epochs × dataset size × model FLOPs.

## 5\. Speed Tips

- Prefer `float32`; avoid `float64` unless necessary.
- Use `basis="rbf"` for slightly fewer ops; `bspline` is closer to paper wording.
- On GPUs, set `device="cuda"` in `train_arkan`; keep batch size such that GPU is saturated.
- DataLoader: `num_workers>0` and `pin_memory=True` if using CUDA.
- Reuse AR memory across configs with the provided cache (already implemented in grid search).
- Avoid very large `p` (> 64) unless needed; costs grow with p² (fit) and p (transform).

## 6\. Profiling

**PyTorch profiler (snippet)**

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU]) as prof:
    with record_function("train_step"):
        pred = model(xb); loss = mse(pred, yb); loss.backward(); opt.step()
print(prof.key_averages().table(sort_by="self_cpu_time_total"))
```

## 7\. Regression Policy

- For PRs, attach before/after MSE on f1 (σ=0.2) and timing over 5 epochs (CPU).
- Performance drops >10% or MSE increases >10% require justification or fixes.
