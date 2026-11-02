#include "h5io.hpp"

#include <hdf5.h>

#include <stdexcept>
#include <vector>

namespace {
void write_vec(hid_t loc, const char* name, const void* data, hsize_t n, hid_t type = H5T_NATIVE_DOUBLE) {
    hid_t space = H5Screate_simple(1, &n, nullptr);
    hid_t ds = H5Dcreate2(loc, name, type == H5T_NATIVE_DOUBLE ? H5T_IEEE_F64LE : H5T_STD_I32LE, space,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(ds);
    H5Sclose(space);
}

void write_attr(hid_t loc, const std::string& name, double v) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t a = H5Acreate2(loc, name.c_str(), H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_DOUBLE, &v);
    H5Aclose(a);
    H5Sclose(space);
}
}  // namespace

Eigen::VectorXd read_series(const std::string& file, const std::string& dataset) {
    hid_t f = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) throw std::runtime_error("cannot open " + file);
    hid_t ds = H5Dopen2(f, dataset.c_str(), H5P_DEFAULT);
    if (ds < 0) {
        H5Fclose(f);
        throw std::runtime_error("dataset not found: " + dataset);
    }
    hid_t space = H5Dget_space(ds);
    hsize_t n = static_cast<hsize_t>(H5Sget_simple_extent_npoints(space));
    Eigen::VectorXd v(n);
    herr_t rc = H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
    H5Sclose(space);
    H5Dclose(ds);
    H5Fclose(f);
    if (rc < 0) throw std::runtime_error("failed to read " + dataset);
    return v;
}

void write_results(const std::string& file, const BacktestResult& r, const BacktestConfig& cfg) {
    hid_t f = H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (f < 0) throw std::runtime_error("cannot create " + file);
    const hsize_t n = r.index.size();
    write_vec(f, "index", r.index.data(), n, H5T_NATIVE_INT);
    write_vec(f, "prediction", r.prediction.data(), n);
    write_vec(f, "position", r.position.data(), n);
    write_vec(f, "realized", r.realized.data(), n);
    write_vec(f, "pnl", r.pnl.data(), n);
    write_vec(f, "equity", r.equity.data(), n);
    for (const auto& [k, v] : r.metrics) write_attr(f, k, v);
    write_attr(f, "train_window", cfg.train_window);
    write_attr(f, "test_window", cfg.test_window);
    write_attr(f, "cost_bps", cfg.cost_bps);
    H5Fclose(f);
}
