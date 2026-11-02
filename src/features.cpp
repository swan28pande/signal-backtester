#include "features.hpp"

#include <cmath>
#include <stdexcept>

Eigen::VectorXd log_returns(const Eigen::VectorXd& close) {
    const Eigen::Index n = close.size();
    Eigen::VectorXd r = Eigen::VectorXd::Zero(n);
    for (Eigen::Index t = 1; t < n; ++t) r[t] = std::log(close[t] / close[t - 1]);
    return r;  // r[t] = return from t-1 to t, r[0] = 0
}

Dataset make_features(const Eigen::VectorXd& close) {
    constexpr int kLags = 5, kWarmup = 21;
    const Eigen::VectorXd r = log_returns(close);
    const Eigen::Index n = close.size();
    if (n < kWarmup + 2) throw std::runtime_error("price series too short");

    Dataset d;
    for (int l = 1; l <= kLags; ++l) d.names.push_back("ret_lag" + std::to_string(l));
    d.names.insert(d.names.end(), {"mom_10", "mom_20", "vol_20", "zscore_20"});

    const Eigen::Index rows = n - 1 - kWarmup;  // last row needs r[t+1] as its target
    d.X.resize(rows, d.names.size());
    d.y.resize(rows);
    for (Eigen::Index i = 0; i < rows; ++i) {
        const Eigen::Index t = i + kWarmup;
        int c = 0;
        for (int l = 0; l < kLags; ++l) d.X(i, c++) = r[t - l];
        d.X(i, c++) = r.segment(t - 9, 10).sum();
        d.X(i, c++) = r.segment(t - 19, 20).sum();
        const Eigen::VectorXd w = r.segment(t - 19, 20);
        d.X(i, c++) = std::sqrt((w.array() - w.mean()).square().mean());
        const Eigen::VectorXd p = close.segment(t - 19, 20);
        const double sd = std::sqrt((p.array() - p.mean()).square().mean());
        d.X(i, c++) = sd > 0 ? (close[t] - p.mean()) / sd : 0.0;
        d.y[i] = r[t + 1];
        d.index.push_back(static_cast<int>(t));
    }
    return d;
}
