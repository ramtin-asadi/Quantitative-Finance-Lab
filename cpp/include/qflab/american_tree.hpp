#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace qflab {

pybind11::array_t<double> american_tree_batch(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> s,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> k,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> r,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> q,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> sigma,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> tau,
    pybind11::array_t<int, pybind11::array::c_style | pybind11::array::forcecast> option_type,
    int steps,
    int tree_type,
    bool american);

pybind11::dict american_tree_boundary(
    double s,
    double k,
    double r,
    double q,
    double sigma,
    double tau,
    int option_type,
    int steps,
    int tree_type,
    bool american);

}  // namespace qflab
