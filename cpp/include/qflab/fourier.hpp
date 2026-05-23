#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace qflab {

pybind11::dict fft_prices(
    int model_id,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> params,
    double s,
    double r,
    double q,
    double tau,
    double alpha,
    int n,
    double eta,
    int option_type);

pybind11::array_t<double> direct_prices(
    int model_id,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> params,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> strikes,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> tau,
    double s,
    double r,
    double q,
    int n_terms,
    double u_max,
    int option_type);

pybind11::array_t<double> cos_prices(
    int model_id,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> params,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> strikes,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> tau,
    double s,
    double r,
    double q,
    int n_terms,
    double truncation_width,
    int option_type);

}  // namespace qflab
