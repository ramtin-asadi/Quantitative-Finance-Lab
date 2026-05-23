#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace qflab {

pybind11::array_t<double> gbm_paths_antithetic(
    double s0,
    double r,
    double q,
    double sigma,
    double tau,
    int steps,
    int paths,
    unsigned int seed);

pybind11::dict lsm_backward(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> paths,
    double strike,
    double r,
    double tau,
    int option_type,
    int degree);

pybind11::dict lsm_eval_policy(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> paths,
    double strike,
    double r,
    double tau,
    int option_type,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> coeffs);

}  // namespace qflab
