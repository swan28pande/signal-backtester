// Predictive signal models: ridge regression (Eigen) and an MLP (LibTorch, optional).
#pragma once
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

class Model {
public:
    virtual ~Model() = default;
    virtual void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) = 0;
    virtual Eigen::VectorXd predict(const Eigen::MatrixXd& X) const = 0;
};

class RidgeModel : public Model {
public:
    explicit RidgeModel(double lambda = 1.0) : lambda_(lambda) {}
    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) override;
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const override;

private:
    double lambda_, intercept_ = 0;
    Eigen::VectorXd w_;
};

// "ridge" or "mlp" (mlp needs a LibTorch build).
std::unique_ptr<Model> make_model(const std::string& name);
std::vector<std::string> available_models();
