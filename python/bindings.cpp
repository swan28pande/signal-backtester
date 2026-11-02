// Python bindings: run the C++ backtester directly on NumPy arrays.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "backtest.hpp"
#include "models.hpp"

namespace py = pybind11;

PYBIND11_MODULE(sigbt, m) {
    m.doc() = "Walk-forward signal backtester (C++17 / Eigen / LibTorch)";
    m.def("available_models", &available_models);
    m.def(
        "run_backtest",
        [](const Eigen::VectorXd& close, const std::string& model, int train, int test, double cost_bps) {
            BacktestConfig cfg{model, train, test, cost_bps};
            BacktestResult r;
            {
                py::gil_scoped_release release;  // the backtest is pure C++
                r = run_backtest(close, cfg);
            }
            py::dict out;
            out["index"] = r.index;
            out["prediction"] = r.prediction;
            out["position"] = r.position;
            out["realized"] = r.realized;
            out["pnl"] = r.pnl;
            out["equity"] = r.equity;
            out["metrics"] = r.metrics;
            out["folds"] = r.folds;
            return out;
        },
        py::arg("close"), py::arg("model") = "ridge", py::arg("train") = 500, py::arg("test") = 50,
        py::arg("cost_bps") = 1.0);
}
