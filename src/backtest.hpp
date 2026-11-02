// Walk-forward validation and P&L simulation.
#pragma once
#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>

struct BacktestConfig {
    std::string model = "ridge";
    int train_window = 500;   // rows used to fit each model
    int test_window = 50;     // rows predicted before refitting
    double cost_bps = 1.0;    // transaction cost per unit of turnover
    double max_position = 1.0;
};

struct BacktestResult {
    std::vector<int> index;               // price index of each out-of-sample row
    Eigen::VectorXd prediction, position, realized, pnl, equity;
    std::map<std::string, double> metrics;
    int folds = 0;
};

BacktestResult run_backtest(const Eigen::VectorXd& close, const BacktestConfig& cfg);
