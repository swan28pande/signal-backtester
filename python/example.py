"""Run the C++ backtester from Python on NumPy data and compare models.

    PYTHONPATH=build python python/example.py prices.h5
"""
import sys

import h5py
import sigbt

close = h5py.File(sys.argv[1] if len(sys.argv) > 1 else "prices.h5")["close"][:]
for model in sigbt.available_models():
    r = sigbt.run_backtest(close, model=model, train=500, test=50, cost_bps=1.0)
    m = r["metrics"]
    print(f"{model:>6}: sharpe {m['sharpe']:5.2f}  IC {m['ic']:+.3f}  "
          f"total {m['total_return']:+.1%}  maxDD {m['max_drawdown']:.1%}  folds {r['folds']}")
