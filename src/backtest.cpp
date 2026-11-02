#include "backtest.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "features.hpp"
#include "models.hpp"

namespace {
double corr(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    const Eigen::ArrayXd da = a.array() - a.mean(), db = b.array() - b.mean();
    const double den = std::sqrt(da.square().sum() * db.square().sum());
    return den > 0 ? (da * db).sum() / den : 0.0;
}
}  // namespace

BacktestResult run_backtest(const Eigen::VectorXd& close, const BacktestConfig& cfg) {
    const Dataset d = make_features(close);
    const int n = static_cast<int>(d.X.rows());
    if (n <= cfg.train_window) throw std::runtime_error("not enough data for one training window");

    const int oos = n - cfg.train_window;
    BacktestResult res;
    res.prediction.resize(oos);
    res.realized = d.y.tail(oos);
    res.index.assign(d.index.end() - oos, d.index.end());

    // Walk forward: fit on [s - train, s), predict [s, s + test), roll by test.
    // Row s-1's target is the return into day s, already known when predicting at s: no lookahead.
    for (int s = cfg.train_window; s < n; s += cfg.test_window) {
        const int len = std::min(cfg.test_window, n - s);
        const Eigen::MatrixXd Xtr = d.X.middleRows(s - cfg.train_window, cfg.train_window);
        const Eigen::RowVectorXd mu = Xtr.colwise().mean();
        Eigen::RowVectorXd sd = ((Xtr.rowwise() - mu).array().square().colwise().mean()).sqrt();
        sd = sd.unaryExpr([](double v) { return v > 1e-12 ? v : 1.0; });
        auto standardize = [&](const Eigen::MatrixXd& X) -> Eigen::MatrixXd {
            return (X.rowwise() - mu).array().rowwise() / sd.array();
        };

        auto model = make_model(cfg.model);
        model->fit(standardize(Xtr), d.y.segment(s - cfg.train_window, cfg.train_window));
        res.prediction.segment(s - cfg.train_window, len) = model->predict(standardize(d.X.middleRows(s, len)));
        ++res.folds;
    }

    // Position sized by prediction strength relative to its recent spread, clipped.
    res.position.resize(oos);
    const double scale = std::max(1e-12, std::sqrt(res.prediction.array().square().mean()));
    for (int i = 0; i < oos; ++i)
        res.position[i] = std::clamp(res.prediction[i] / scale, -cfg.max_position, cfg.max_position);

    res.pnl.resize(oos);
    res.equity.resize(oos);
    double prev_pos = 0, eq = 0, peak = 0, max_dd = 0, turnover = 0;
    for (int i = 0; i < oos; ++i) {
        const double trade = std::abs(res.position[i] - prev_pos);
        turnover += trade;
        res.pnl[i] = res.position[i] * res.realized[i] - trade * cfg.cost_bps * 1e-4;
        prev_pos = res.position[i];
        eq += res.pnl[i];
        res.equity[i] = eq;  // cumulative log P&L
        peak = std::max(peak, eq);
        max_dd = std::max(max_dd, peak - eq);
    }

    const double mean = res.pnl.mean();
    const double sd = std::sqrt((res.pnl.array() - mean).square().mean());
    int hits = 0, active = 0;
    for (int i = 0; i < oos; ++i) {
        if (res.prediction[i] == 0) continue;
        ++active;
        hits += (res.prediction[i] > 0) == (res.realized[i] > 0);
    }
    res.metrics = {
        {"total_return", std::exp(eq) - 1},
        {"annual_return", mean * 252},
        {"sharpe", sd > 0 ? mean / sd * std::sqrt(252.0) : 0.0},
        {"max_drawdown", 1 - std::exp(-max_dd)},
        {"hit_rate", active ? double(hits) / active : 0.0},
        {"ic", corr(res.prediction, res.realized)},
        {"avg_daily_turnover", turnover / oos},
        {"buy_hold_return", std::exp(res.realized.sum()) - 1},
    };
    return res;
}
