// Feature engineering on a single price series (Eigen).
#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>

struct Dataset {
    Eigen::MatrixXd X;          // row t: features known at the close of day t
    Eigen::VectorXd y;          // row t: return from t to t+1 (the target)
    std::vector<int> index;     // row t -> position in the price series
    std::vector<std::string> names;
};

Eigen::VectorXd log_returns(const Eigen::VectorXd& close);

// Lagged returns, momentum, volatility and a moving-average z-score.
Dataset make_features(const Eigen::VectorXd& close);
