// backtest: walk-forward backtest of a predictive signal on an HDF5 price series.
//
//   backtest --data prices.h5 [--dataset /close] [--model ridge|mlp]
//            [--train 500] [--test 50] [--cost-bps 1] [--out results.h5]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "backtest.hpp"
#include "h5io.hpp"
#include "models.hpp"

int main(int argc, char** argv) {
    std::string data, dataset = "/close", out;
    BacktestConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--data") data = next();
        else if (a == "--dataset") dataset = next();
        else if (a == "--model") cfg.model = next();
        else if (a == "--train") cfg.train_window = std::atoi(next().c_str());
        else if (a == "--test") cfg.test_window = std::atoi(next().c_str());
        else if (a == "--cost-bps") cfg.cost_bps = std::atof(next().c_str());
        else if (a == "--out") out = next();
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }
    if (data.empty()) {
        std::string models;
        for (auto& m : available_models()) models += " " + m;
        std::fprintf(stderr, "usage: backtest --data prices.h5 [--model ridge|mlp] ...  (models built:%s)\n", models.c_str());
        return 1;
    }

    try {
        const auto close = read_series(data, dataset);
        const auto t0 = std::chrono::steady_clock::now();
        const auto r = run_backtest(close, cfg);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        std::printf("model=%s  prices=%ld  out-of-sample days=%zu  folds=%d  (%.2fs)\n", cfg.model.c_str(),
                    long(close.size()), r.index.size(), r.folds, secs);
        auto m = r.metrics;
        std::printf("  total return     %+8.2f%%   (buy & hold %+.2f%%)\n", 100 * m["total_return"], 100 * m["buy_hold_return"]);
        std::printf("  annual return    %+8.2f%%\n", 100 * m["annual_return"]);
        std::printf("  sharpe           %8.2f\n", m["sharpe"]);
        std::printf("  max drawdown     %8.2f%%\n", 100 * m["max_drawdown"]);
        std::printf("  hit rate         %8.2f%%\n", 100 * m["hit_rate"]);
        std::printf("  IC               %8.3f\n", m["ic"]);
        std::printf("  daily turnover   %8.2f\n", m["avg_daily_turnover"]);
        if (!out.empty()) {
            write_results(out, r, cfg);
            std::printf("results written to %s\n", out.c_str());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
