#include <pybind11/pybind11.h>

#include "qflab/american_lsm.hpp"
#include "qflab/american_pde.hpp"
#include "qflab/american_tree.hpp"
#include "qflab/fourier.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_kernels, m) {
    m.doc() = "Compiled numerical kernels for QuantFinLab option projects";

    m.def("american_tree_batch", &qflab::american_tree_batch,
          py::arg("s"), py::arg("k"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("tau"),
          py::arg("option_type"), py::arg("steps") = 200, py::arg("tree_type") = 0, py::arg("american") = true);
    m.def("american_tree_boundary", &qflab::american_tree_boundary,
          py::arg("s"), py::arg("k"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("tau"),
          py::arg("option_type"), py::arg("steps") = 200, py::arg("tree_type") = 0, py::arg("american") = true);
    m.def("american_pde_psor", &qflab::american_pde_psor,
          py::arg("s"), py::arg("k"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("tau"),
          py::arg("option_type"), py::arg("s_steps") = 160, py::arg("t_steps") = 120,
          py::arg("s_max_mult") = 3.0, py::arg("omega") = 1.35, py::arg("tol") = 1e-7,
          py::arg("max_iter") = 5000, py::arg("american") = true);
    m.def("american_pde_psor_batch", &qflab::american_pde_psor_batch,
          py::arg("s"), py::arg("k"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("tau"),
          py::arg("option_type"), py::arg("s_steps") = 100, py::arg("t_steps") = 80,
          py::arg("s_max_mult") = 3.0, py::arg("omega") = 1.35, py::arg("tol") = 1e-7,
          py::arg("max_iter") = 3000, py::arg("american") = true);
    m.def("gbm_paths_antithetic", &qflab::gbm_paths_antithetic,
          py::arg("s0"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("tau"),
          py::arg("steps"), py::arg("paths"), py::arg("seed") = 7);
    m.def("lsm_backward", &qflab::lsm_backward,
          py::arg("paths"), py::arg("strike"), py::arg("r"), py::arg("tau"), py::arg("option_type"), py::arg("degree") = 3);
    m.def("lsm_eval_policy", &qflab::lsm_eval_policy,
          py::arg("paths"), py::arg("strike"), py::arg("r"), py::arg("tau"), py::arg("option_type"), py::arg("coefficients"));
    m.def("fft_prices", &qflab::fft_prices,
          py::arg("model_id"), py::arg("params"), py::arg("s"), py::arg("r"), py::arg("q"), py::arg("tau"),
          py::arg("alpha") = 1.5, py::arg("n") = 256, py::arg("eta") = 0.25, py::arg("option_type") = 1);
    m.def("direct_prices", &qflab::direct_prices,
          py::arg("model_id"), py::arg("params"), py::arg("strikes"), py::arg("tau"), py::arg("s"),
          py::arg("r"), py::arg("q"), py::arg("n_terms") = 512, py::arg("u_max") = 120.0,
          py::arg("option_type") = 1);
    m.def("cos_prices", &qflab::cos_prices,
          py::arg("model_id"), py::arg("params"), py::arg("strikes"), py::arg("tau"), py::arg("s"),
          py::arg("r"), py::arg("q"), py::arg("n_terms") = 256, py::arg("truncation_width") = 100.0,
          py::arg("option_type") = 1);
}
