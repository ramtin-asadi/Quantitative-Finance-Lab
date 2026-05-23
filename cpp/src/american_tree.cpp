#include "qflab/american_tree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace qflab {
namespace {

double payoff(double s, double k, int option_type) {
    if (option_type > 0) {
        return std::max(s - k, 0.0);
    }
    return std::max(k - s, 0.0);
}

void tree_ud_p(double r, double q, double sigma, double dt, int tree_type, double& u, double& d, double& p) {
    const double drift = std::exp((r - q) * dt);
    if (tree_type == 1) {
        const double v = std::exp(sigma * sigma * dt);
        const double root = std::sqrt(std::max(v * v + 2.0 * v - 3.0, 0.0));
        u = 0.5 * drift * v * (v + 1.0 + root);
        d = 0.5 * drift * v * (v + 1.0 - root);
        if (!(u > d) || !std::isfinite(u) || !std::isfinite(d)) {
            u = std::exp(sigma * std::sqrt(dt));
            d = 1.0 / u;
        }
    } else {
        u = std::exp(sigma * std::sqrt(dt));
        d = 1.0 / u;
    }
    p = (drift - d) / (u - d);
    p = std::min(std::max(p, 1e-10), 1.0 - 1e-10);
}

double tree_price_one(double s, double k, double r, double q, double sigma, double tau, int option_type, int steps, int tree_type, bool american) {
    if (!(s > 0.0) || !(k > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (tau <= 0.0 || sigma <= 0.0 || steps <= 1) {
        return payoff(s, k, option_type);
    }
    const int n = std::max(2, steps);
    const double dt = tau / static_cast<double>(n);
    double u, d, p;
    tree_ud_p(r, q, sigma, dt, tree_type, u, d, p);
    const double disc = std::exp(-r * dt);
    std::vector<double> v(n + 1);
    const double ratio = u / d;
    double st = s * std::exp(static_cast<double>(n) * std::log(d));
    for (int j = 0; j <= n; ++j) {
        v[j] = payoff(st, k, option_type);
        st *= ratio;
    }
    for (int i = n - 1; i >= 0; --i) {
        double node_s = s * std::exp(static_cast<double>(i) * std::log(d));
        for (int j = 0; j <= i; ++j) {
            const double cont = disc * (p * v[j + 1] + (1.0 - p) * v[j]);
            if (american) {
                v[j] = std::max(payoff(node_s, k, option_type), cont);
            } else {
                v[j] = cont;
            }
            node_s *= ratio;
        }
    }
    return v[0];
}

}  // namespace

py::array_t<double> american_tree_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> s,
    py::array_t<double, py::array::c_style | py::array::forcecast> k,
    py::array_t<double, py::array::c_style | py::array::forcecast> r,
    py::array_t<double, py::array::c_style | py::array::forcecast> q,
    py::array_t<double, py::array::c_style | py::array::forcecast> sigma,
    py::array_t<double, py::array::c_style | py::array::forcecast> tau,
    py::array_t<int, py::array::c_style | py::array::forcecast> option_type,
    int steps,
    int tree_type,
    bool american) {
    const auto bs = s.request();
    const auto bk = k.request();
    const auto br = r.request();
    const auto bq = q.request();
    const auto bv = sigma.request();
    const auto bt = tau.request();
    const auto bo = option_type.request();
    const py::ssize_t n = bs.size;
    if (bk.size != n || br.size != n || bq.size != n || bv.size != n || bt.size != n || bo.size != n) {
        throw std::runtime_error("all input arrays must have the same length");
    }
    auto out = py::array_t<double>(n);
    auto bout = out.mutable_unchecked<1>();
    const auto xs = s.unchecked<1>();
    const auto xk = k.unchecked<1>();
    const auto xr = r.unchecked<1>();
    const auto xq = q.unchecked<1>();
    const auto xv = sigma.unchecked<1>();
    const auto xt = tau.unchecked<1>();
    const auto xo = option_type.unchecked<1>();
    {
        py::gil_scoped_release release;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 64)
        #endif
        for (py::ssize_t idx = 0; idx < n; ++idx) {
            bout(idx) = tree_price_one(xs(idx), xk(idx), xr(idx), xq(idx), xv(idx), xt(idx), xo(idx), steps, tree_type, american);
        }
    }
    return out;
}

py::dict american_tree_boundary(
    double s,
    double k,
    double r,
    double q,
    double sigma,
    double tau,
    int option_type,
    int steps,
    int tree_type,
    bool american) {
    const int n = std::max(2, steps);
    auto times = py::array_t<double>(n + 1);
    auto boundary = py::array_t<double>(n + 1);
    auto t = times.mutable_unchecked<1>();
    auto b = boundary.mutable_unchecked<1>();
    for (int i = 0; i <= n; ++i) {
        t(i) = tau * static_cast<double>(i) / static_cast<double>(n);
        b(i) = std::numeric_limits<double>::quiet_NaN();
    }
    if (!(s > 0.0) || !(k > 0.0) || tau <= 0.0 || sigma <= 0.0) {
        py::dict out;
        out["times"] = times;
        out["boundary"] = boundary;
        out["price"] = payoff(s, k, option_type);
        return out;
    }
    const double dt = tau / static_cast<double>(n);
    double u, d, p;
    tree_ud_p(r, q, sigma, dt, tree_type, u, d, p);
    const double disc = std::exp(-r * dt);
    std::vector<double> v(n + 1);
    const double ratio = u / d;
    double st_terminal = s * std::exp(static_cast<double>(n) * std::log(d));
    for (int j = 0; j <= n; ++j) {
        v[j] = payoff(st_terminal, k, option_type);
        st_terminal *= ratio;
    }
    b(n) = k;
    for (int i = n - 1; i >= 0; --i) {
        double level = std::numeric_limits<double>::quiet_NaN();
        double st = s * std::exp(static_cast<double>(i) * std::log(d));
        for (int j = 0; j <= i; ++j) {
            const double cont = disc * (p * v[j + 1] + (1.0 - p) * v[j]);
            const double ex = payoff(st, k, option_type);
            const bool bind = american && ex > cont + 1e-10;
            if (bind) {
                if (option_type > 0) {
                    if (!std::isfinite(level) || st < level) {
                        level = st;
                    }
                } else {
                    if (!std::isfinite(level) || st > level) {
                        level = st;
                    }
                }
            }
            v[j] = american ? std::max(ex, cont) : cont;
            st *= ratio;
        }
        b(i) = level;
    }
    py::dict out;
    out["times"] = times;
    out["boundary"] = boundary;
    out["price"] = v[0];
    return out;
}

}  // namespace qflab
