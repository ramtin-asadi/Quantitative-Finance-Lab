#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace qflab {

pybind11::dict american_pde_psor(
    double s,
    double k,
    double r,
    double q,
    double sigma,
    double tau,
    int option_type,
    int s_steps,
    int t_steps,
    double s_max_mult,
    double omega,
    double tol,
    int max_iter,
    bool american);

pybind11::dict american_pde_psor_batch(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> s,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> k,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> r,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> q,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> sigma,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> tau,
    pybind11::array_t<int, pybind11::array::c_style | pybind11::array::forcecast> option_type,
    int s_steps,
    int t_steps,
    double s_max_mult,
    double omega,
    double tol,
    int max_iter,
    bool american);

}  // namespace qflab
