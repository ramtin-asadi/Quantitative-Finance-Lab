#include "qflab/american_pde.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace qflab {
namespace {

double payoff_pde(double s, double k, int option_type) {
    return option_type > 0 ? std::max(s - k, 0.0) : std::max(k - s, 0.0);
}

py::dict pde_one(
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
    bool american,
    bool keep_grid) {
    const int m = std::max(20, s_steps);
    const int n = std::max(4, t_steps);
    const double s_max = std::max(s_max_mult * std::max(s, k), 1.5 * std::max(s, k));
    const double ds = s_max / static_cast<double>(m);
    const double dt = std::max(tau, 1e-10) / static_cast<double>(n);

    std::vector<double> grid(m + 1), oldv(m + 1), newv(m + 1), rhs(m + 1), pay(m + 1);
    for (int i = 0; i <= m; ++i) {
        grid[i] = i * ds;
        pay[i] = payoff_pde(grid[i], k, option_type);
        oldv[i] = pay[i];
        newv[i] = pay[i];
    }

    auto boundary = py::array_t<double>(n + 1);
    auto residuals = py::array_t<double>(n);
    auto b = boundary.mutable_unchecked<1>();
    auto res = residuals.mutable_unchecked<1>();
    for (int j = 0; j <= n; ++j) {
        b(j) = std::numeric_limits<double>::quiet_NaN();
    }
    b(n) = k;

    py::array_t<double> values;
    if (keep_grid) {
        values = py::array_t<double>({n + 1, m + 1});
        auto vg = values.mutable_unchecked<2>();
        for (int i = 0; i <= m; ++i) {
            vg(n, i) = oldv[i];
        }
    }

    const double sig2 = sigma * sigma;
    for (int step = n - 1; step >= 0; --step) {
        const double time = step * dt;
        if (option_type > 0) {
            rhs[0] = 0.0;
            rhs[m] = s_max * std::exp(-q * (tau - time)) - k * std::exp(-r * (tau - time));
        } else {
            rhs[0] = k * std::exp(-r * (tau - time));
            rhs[m] = 0.0;
        }
        rhs[0] = std::max(rhs[0], pay[0]);
        rhs[m] = std::max(rhs[m], pay[m]);
        for (int i = 1; i < m; ++i) {
            rhs[i] = oldv[i];
            newv[i] = oldv[i];
        }
        newv[0] = rhs[0];
        newv[m] = rhs[m];

        double max_update = 0.0;
        for (int it = 0; it < max_iter; ++it) {
            max_update = 0.0;
            for (int i = 1; i < m; ++i) {
                const double ii = static_cast<double>(i);
                const double a = -0.5 * dt * (sig2 * ii * ii - (r - q) * ii);
                const double c = -0.5 * dt * (sig2 * ii * ii + (r - q) * ii);
                const double diag = 1.0 + dt * (sig2 * ii * ii + r);
                const double y = (rhs[i] - a * newv[i - 1] - c * newv[i + 1]) / diag;
                double candidate = newv[i] + omega * (y - newv[i]);
                if (american) {
                    candidate = std::max(candidate, pay[i]);
                }
                max_update = std::max(max_update, std::abs(candidate - newv[i]));
                newv[i] = candidate;
            }
            if (max_update < tol) {
                break;
            }
        }
        res(step) = max_update;

        double level = std::numeric_limits<double>::quiet_NaN();
        for (int i = 1; i < m; ++i) {
            if (american && std::abs(newv[i] - pay[i]) < 5e-5 && pay[i] > 0.0) {
                if (option_type > 0) {
                    if (!std::isfinite(level) || grid[i] < level) {
                        level = grid[i];
                    }
                } else {
                    if (!std::isfinite(level) || grid[i] > level) {
                        level = grid[i];
                    }
                }
            }
        }
        b(step) = level;
        oldv.swap(newv);
        if (keep_grid) {
            auto vg = values.mutable_unchecked<2>();
            for (int i = 0; i <= m; ++i) {
                vg(step, i) = oldv[i];
            }
        }
    }

    int j = static_cast<int>(std::floor(s / ds));
    j = std::min(std::max(j, 0), m - 1);
    const double w = (s - grid[j]) / ds;
    const double price = oldv[j] * (1.0 - w) + oldv[j + 1] * w;

    auto s_grid = py::array_t<double>(m + 1);
    auto sg = s_grid.mutable_unchecked<1>();
    for (int i = 0; i <= m; ++i) {
        sg(i) = grid[i];
    }

    py::dict out;
    out["price"] = price;
    out["s_grid"] = s_grid;
    out["boundary"] = boundary;
    out["residuals"] = residuals;
    if (keep_grid) {
        out["values"] = values;
    }
    return out;
}

}  // namespace

py::dict american_pde_psor(
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
    bool american) {
    return pde_one(s, k, r, q, sigma, tau, option_type, s_steps, t_steps, s_max_mult, omega, tol, max_iter, american, true);
}

py::dict american_pde_psor_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> s,
    py::array_t<double, py::array::c_style | py::array::forcecast> k,
    py::array_t<double, py::array::c_style | py::array::forcecast> r,
    py::array_t<double, py::array::c_style | py::array::forcecast> q,
    py::array_t<double, py::array::c_style | py::array::forcecast> sigma,
    py::array_t<double, py::array::c_style | py::array::forcecast> tau,
    py::array_t<int, py::array::c_style | py::array::forcecast> option_type,
    int s_steps,
    int t_steps,
    double s_max_mult,
    double omega,
    double tol,
    int max_iter,
    bool american) {
    const auto bs = s.request();
    const py::ssize_t n = bs.size;
    if (k.request().size != n || r.request().size != n || q.request().size != n || sigma.request().size != n || tau.request().size != n || option_type.request().size != n) {
        throw std::runtime_error("all input arrays must have the same length");
    }
    auto prices = py::array_t<double>(n);
    auto residual = py::array_t<double>(n);
    auto px = prices.mutable_unchecked<1>();
    auto er = residual.mutable_unchecked<1>();
    const auto xs = s.unchecked<1>();
    const auto xk = k.unchecked<1>();
    const auto xr = r.unchecked<1>();
    const auto xq = q.unchecked<1>();
    const auto xv = sigma.unchecked<1>();
    const auto xt = tau.unchecked<1>();
    const auto xo = option_type.unchecked<1>();
    for (py::ssize_t idx = 0; idx < n; ++idx) {
        py::dict one = pde_one(xs(idx), xk(idx), xr(idx), xq(idx), xv(idx), xt(idx), xo(idx), s_steps, t_steps, s_max_mult, omega, tol, max_iter, american, false);
        px(idx) = one["price"].cast<double>();
        auto res = one["residuals"].cast<py::array_t<double>>();
        auto rr = res.unchecked<1>();
        double max_res = 0.0;
        for (py::ssize_t j = 0; j < rr.shape(0); ++j) {
            max_res = std::max(max_res, rr(j));
        }
        er(idx) = max_res;
    }
    py::dict out;
    out["prices"] = prices;
    out["residuals"] = residual;
    return out;
}

}  // namespace qflab
