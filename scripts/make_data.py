"""Write a synthetic price series to HDF5 (dataset /close).

Daily returns follow a regime-switching AR(1): short-term momentum in some
regimes, mean reversion in others, plus a small nonlinear volatility effect,
so there is a weak but learnable signal. Seeded for reproducibility.

    python scripts/make_data.py prices.h5 --days 3000
"""
import argparse

import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("out", nargs="?", default="prices.h5")
p.add_argument("--days", type=int, default=3000)
p.add_argument("--seed", type=int, default=7)
a = p.parse_args()

rng = np.random.default_rng(a.seed)
phi = np.repeat(rng.choice([0.12, -0.10], size=a.days // 500 + 1), 500)[: a.days]
r = np.zeros(a.days)
for t in range(1, a.days):
    vol = 0.010 * (1.5 if abs(r[t - 1]) > 0.02 else 1.0)
    r[t] = 0.0002 + phi[t] * r[t - 1] + vol * rng.standard_normal()
close = 100 * np.exp(np.cumsum(r))

with h5py.File(a.out, "w") as f:
    f.create_dataset("close", data=close)
    f["close"].attrs["description"] = "synthetic daily close, regime-switching AR(1) returns"
print(f"wrote {a.days} prices to {a.out}")
