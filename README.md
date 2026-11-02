# Predictive Signal Backtester

A C++17 engine that trains predictive signal models on time-series data and evaluates them with **walk-forward validation** and **P&L simulation**. It reads and writes HDF5, runs models in Eigen (ridge) and LibTorch (MLP), and is usable from Python through pybind11.

Stack: C++17, LibTorch, Eigen, pybind11, HDF5.

## How it works

```
prices.h5 ─► features (Eigen) ─► walk-forward loop ─────────────────────► P&L sim ─► metrics + results.h5
  /close      lags, momentum,     fit on [s-train, s)  (ridge | MLP)        position, costs,
              vol, MA z-score     predict [s, s+test), roll by test         equity, drawdown
```

- **Features** (`src/features.cpp`), all known at the close of day *t*:
  - 5 lagged log returns
  - 10- and 20-day momentum
  - 20-day volatility
  - 20-day moving-average z-score

  The target is the return from *t* to *t+1*.
- **Walk-forward** (`src/backtest.cpp`): for each fold, the model is refit on the trailing `train` rows. Features are standardized with that window's statistics only, then the next `test` rows are predicted. There is no lookahead: every training target is already observed when the fold starts.
- **Models** (`src/models.cpp`):
  - `ridge`: closed-form Eigen LDLT solve.
  - `mlp`: 2-layer LibTorch network trained with Adam. It is compiled in only when LibTorch is found.
- **P&L**:
  - position = prediction ÷ its RMS, clipped to ±1
  - P&L = position × next return − turnover × cost
  - Metrics: total and annual return, Sharpe, max drawdown, hit rate, IC (prediction/return correlation), turnover, and buy & hold for comparison.

## Build

```bash
# Debian/Ubuntu: apt install g++ cmake libeigen3-dev libhdf5-dev
pip install numpy h5py pybind11 torch

cmake -S . -B build \
  -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") \
  -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake --build build -j
```

LibTorch and pybind11 are optional: without them you get the ridge model and the CLI only. A standalone LibTorch download works too; point `CMAKE_PREFIX_PATH` at it.

## Usage

```bash
python scripts/make_data.py prices.h5                 # synthetic regime-switching series
./build/backtest --data prices.h5 --model ridge --train 500 --test 50 --cost-bps 1 --out results.h5
./build/backtest --data prices.h5 --model mlp
PYTHONPATH=build python python/example.py prices.h5   # same engine from Python / NumPy
```

`results.h5` holds `index`, `prediction`, `position`, `realized`, `pnl`, and `equity` datasets, with the metrics stored as root attributes.

```python
import sigbt, h5py
r = sigbt.run_backtest(h5py.File("prices.h5")["close"][:], model="ridge", train=500, test=50)
r["metrics"]["sharpe"], r["equity"]          # NumPy arrays back
```

## Sample results

3,000 synthetic days, 1 bp costs, Linux container (CPU):

| Model | Train / test | Folds | Sharpe | IC | Total return | Buy & hold | Time |
|---|---|---|---|---|---|---|---|
| ridge | 500 / 50 | 50 | 0.31 | 0.029 | +43.2% | +42.8% | <0.01 s |
| mlp | 500 / 50 | 50 | −0.36 | 0.004 | −33.6% | +42.8% | 16.4 s |
| ridge | 250 / 20 | 137 | 0.27 | 0.029 | +40.3% | +16.1% | <0.01 s |
| mlp | 250 / 20 | 137 | 0.03 | 0.010 | +4.0% | +16.1% | 29.4 s |

On a weak, regime-switching signal, the regularized linear model generalizes better out of sample than the MLP, which overfits each training window. This is the kind of result walk-forward validation exists to catch.

## Status

Early skeleton:
- Single-asset series only.
- Simple position sizing.
- No hyperparameter search inside folds.
