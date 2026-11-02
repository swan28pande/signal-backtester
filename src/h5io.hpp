// Minimal HDF5 I/O (C API): read a 1-D price dataset, write backtest results.
#pragma once
#include <Eigen/Dense>
#include <string>

#include "backtest.hpp"

Eigen::VectorXd read_series(const std::string& file, const std::string& dataset);
void write_results(const std::string& file, const BacktestResult& r, const BacktestConfig& cfg);
