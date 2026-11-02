#include "models.hpp"

#include <stdexcept>

#ifdef SB_WITH_TORCH
#include <torch/torch.h>
#endif

// Features arrive standardized, so the penalty treats them equally; the intercept is not penalized.
void RidgeModel::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    intercept_ = y.mean();
    const Eigen::MatrixXd A =
        X.transpose() * X + lambda_ * X.rows() * Eigen::MatrixXd::Identity(X.cols(), X.cols());
    w_ = A.ldlt().solve(X.transpose() * (y.array() - intercept_).matrix());
}

Eigen::VectorXd RidgeModel::predict(const Eigen::MatrixXd& X) const {
    return (X * w_).array() + intercept_;
}

#ifdef SB_WITH_TORCH
namespace {
torch::Tensor to_tensor(const Eigen::MatrixXd& m) {
    // Eigen is column-major: view as (cols, rows) and transpose.
    auto t = torch::from_blob(const_cast<double*>(m.data()), {m.cols(), m.rows()}, torch::kFloat64);
    return t.t().to(torch::kFloat32).contiguous();
}

class MlpModel : public Model {
public:
    MlpModel(int hidden = 32, int epochs = 300, double lr = 1e-3) : hidden_(hidden), epochs_(epochs), lr_(lr) {}

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) override {
        torch::manual_seed(0);
        net_ = torch::nn::Sequential(torch::nn::Linear(X.cols(), hidden_), torch::nn::ReLU(),
                                     torch::nn::Linear(hidden_, 1));
        // Returns are tiny; scale the target so the loss is well conditioned.
        y_scale_ = std::max(1e-8, std::sqrt((y.array() - y.mean()).square().mean()));
        auto xt = to_tensor(X);
        auto yt = to_tensor(y / y_scale_);
        torch::optim::Adam opt(net_->parameters(), torch::optim::AdamOptions(lr_).weight_decay(1e-3));
        for (int e = 0; e < epochs_; ++e) {
            opt.zero_grad();
            auto loss = torch::mse_loss(net_->forward(xt), yt);
            loss.backward();
            opt.step();
        }
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const override {
        torch::NoGradGuard ng;
        auto out = net_->forward(to_tensor(X)).to(torch::kFloat64).contiguous();
        return Eigen::Map<Eigen::VectorXd>(out.data_ptr<double>(), X.rows()) * y_scale_;
    }

private:
    int hidden_, epochs_;
    double lr_, y_scale_ = 1.0;
    mutable torch::nn::Sequential net_{nullptr};
};
}  // namespace
#endif

std::unique_ptr<Model> make_model(const std::string& name) {
    if (name == "ridge") return std::make_unique<RidgeModel>(0.1);
#ifdef SB_WITH_TORCH
    if (name == "mlp") return std::make_unique<MlpModel>();
#endif
    throw std::runtime_error("unknown or unavailable model: " + name);
}

std::vector<std::string> available_models() {
#ifdef SB_WITH_TORCH
    return {"ridge", "mlp"};
#else
    return {"ridge"};
#endif
}
