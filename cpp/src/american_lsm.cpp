#include "qflab/american_lsm.hpp"

#include "qflab/rng.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace qflab {
namespace {

double payoff_lsm(double s, double k, int option_type) {
    return option_type > 0 ? std::max(s - k, 0.0) : std::max(k - s, 0.0);
}

std::vector<double> solve_linear(std::vector<double> a, std::vector<double> b, int n) {
    for (int p = 0; p < n; ++p) {
        int pivot = p;
        double best = std::abs(a[p * n + p]);
        for (int i = p + 1; i < n; ++i) {
            const double val = std::abs(a[i * n + p]);
            if (val > best) {
                best = val;
                pivot = i;
            }
        }
        if (best < 1e-12) {
            continue;
        }
        if (pivot != p) {
            for (int j = p; j < n; ++j) {
                std::swap(a[p * n + j], a[pivot * n + j]);
            }
            std::swap(b[p], b[pivot]);
        }
        const double diag = a[p * n + p];
        for (int j = p; j < n; ++j) {
            a[p * n + j] /= diag;
        }
        b[p] /= diag;
        for (int i = 0; i < n; ++i) {
            if (i == p) {
                continue;
            }
            const double f = a[i * n + p];
            for (int j = p; j < n; ++j) {
                a[i * n + j] -= f * a[p * n + j];
            }
            b[i] -= f * b[p];
        }
    }
    return b;
}

double eval_poly(const std::vector<double>& beta, double x) {
    double out = 0.0;
    double p = 1.0;
    for (double v : beta) {
        out += v * p;
        p *= x;
    }
    return out;
}

}  // namespace

py::array_t<double> gbm_paths_antithetic(double s0, double r, double q, double sigma, double tau, int steps, int paths, unsigned int seed) {
    const int n_steps = std::max(1, steps);
    const int n_half = std::max(1, paths / 2);
    const int n_paths = 2 * n_half;
    auto out = py::array_t<double>({n_paths, n_steps + 1});
    auto x = out.mutable_unchecked<2>();
    std::mt19937 rng(seed);
    const double dt = tau / static_cast<double>(n_steps);
    const double drift = (r - q - 0.5 * sigma * sigma) * dt;
    const double vol = sigma * std::sqrt(dt);
    for (int p = 0; p < n_half; ++p) {
        x(p, 0) = s0;
        x(p + n_half, 0) = s0;
        for (int j = 1; j <= n_steps; ++j) {
            const double z = standard_normal(rng);
            x(p, j) = x(p, j - 1) * std::exp(drift + vol * z);
            x(p + n_half, j) = x(p + n_half, j - 1) * std::exp(drift - vol * z);
        }
    }
    return out;
}

py::dict lsm_backward(
    py::array_t<double, py::array::c_style | py::array::forcecast> paths,
    double strike,
    double r,
    double tau,
    int option_type,
    int degree) {
    const auto b = paths.request();
    if (b.ndim != 2) {
        throw std::runtime_error("paths must be a 2D array");
    }
    const int n_paths = static_cast<int>(b.shape[0]);
    const int n_steps = static_cast<int>(b.shape[1]) - 1;
    const int d = std::max(0, degree);
    const int pcols = d + 1;
    const double dt = tau / std::max(n_steps, 1);
    const double disc = std::exp(-r * dt);
    const auto x = paths.unchecked<2>();

    std::vector<double> cf(n_paths, 0.0);
    std::vector<int> exercise(n_paths, n_steps);
    for (int i = 0; i < n_paths; ++i) {
        cf[i] = payoff_lsm(x(i, n_steps), strike, option_type);
    }

    auto coeffs = py::array_t<double>({n_steps + 1, pcols});
    auto c = coeffs.mutable_unchecked<2>();
    for (int t = 0; t <= n_steps; ++t) {
        for (int j = 0; j < pcols; ++j) {
            c(t, j) = 0.0;
        }
    }

    for (int t = n_steps - 1; t >= 1; --t) {
        for (int i = 0; i < n_paths; ++i) {
            cf[i] *= disc;
        }
        std::vector<double> ata(pcols * pcols, 0.0), aty(pcols, 0.0);
        int n_itm = 0;
        for (int i = 0; i < n_paths; ++i) {
            const double ex = payoff_lsm(x(i, t), strike, option_type);
            if (ex <= 0.0) {
                continue;
            }
            ++n_itm;
            const double z = std::log(std::max(x(i, t), 1e-300) / strike);
            std::vector<double> basis(pcols, 1.0);
            for (int j = 1; j < pcols; ++j) {
                basis[j] = basis[j - 1] * z;
            }
            for (int a = 0; a < pcols; ++a) {
                aty[a] += basis[a] * cf[i];
                for (int bb = 0; bb < pcols; ++bb) {
                    ata[a * pcols + bb] += basis[a] * basis[bb];
                }
            }
        }
        std::vector<double> beta(pcols, 0.0);
        if (n_itm >= pcols) {
            for (int j = 0; j < pcols; ++j) {
                ata[j * pcols + j] += 1e-10;
            }
            beta = solve_linear(ata, aty, pcols);
        }
        for (int j = 0; j < pcols; ++j) {
            c(t, j) = beta[j];
        }
        for (int i = 0; i < n_paths; ++i) {
            const double ex = payoff_lsm(x(i, t), strike, option_type);
            if (ex <= 0.0) {
                continue;
            }
            const double z = std::log(std::max(x(i, t), 1e-300) / strike);
            const double cont = eval_poly(beta, z);
            if (ex > cont) {
                cf[i] = ex;
                exercise[i] = t;
            }
        }
    }
    for (int i = 0; i < n_paths; ++i) {
        cf[i] *= disc;
    }
    double price = 0.0;
    for (double y : cf) {
        price += y;
    }
    price /= std::max(n_paths, 1);

    auto ex_arr = py::array_t<int>(n_paths);
    auto e = ex_arr.mutable_unchecked<1>();
    for (int i = 0; i < n_paths; ++i) {
        e(i) = exercise[i];
    }
    py::dict out;
    out["price"] = price;
    out["exercise_time"] = ex_arr;
    out["coefficients"] = coeffs;
    return out;
}

py::dict lsm_eval_policy(
    py::array_t<double, py::array::c_style | py::array::forcecast> paths,
    double strike,
    double r,
    double tau,
    int option_type,
    py::array_t<double, py::array::c_style | py::array::forcecast> coeffs) {
    const auto bp = paths.request();
    const auto bc = coeffs.request();
    if (bp.ndim != 2 || bc.ndim != 2) {
        throw std::runtime_error("paths and coefficients must be 2D arrays");
    }
    const int n_paths = static_cast<int>(bp.shape[0]);
    const int n_steps = static_cast<int>(bp.shape[1]) - 1;
    const int pcols = static_cast<int>(bc.shape[1]);
    const double dt = tau / std::max(n_steps, 1);
    const auto x = paths.unchecked<2>();
    const auto c = coeffs.unchecked<2>();
    auto ex_arr = py::array_t<int>(n_paths);
    auto e = ex_arr.mutable_unchecked<1>();
    double sum = 0.0;
    for (int i = 0; i < n_paths; ++i) {
        double cash = payoff_lsm(x(i, n_steps), strike, option_type) * std::exp(-r * tau);
        int et = n_steps;
        for (int t = 1; t < n_steps; ++t) {
            const double ex = payoff_lsm(x(i, t), strike, option_type);
            if (ex <= 0.0) {
                continue;
            }
            std::vector<double> beta(pcols, 0.0);
            for (int j = 0; j < pcols; ++j) {
                beta[j] = c(t, j);
            }
            const double z = std::log(std::max(x(i, t), 1e-300) / strike);
            const double cont = eval_poly(beta, z);
            if (ex > cont) {
                cash = ex * std::exp(-r * dt * static_cast<double>(t));
                et = t;
                break;
            }
        }
        sum += cash;
        e(i) = et;
    }
    py::dict out;
    out["price"] = sum / std::max(n_paths, 1);
    out["exercise_time"] = ex_arr;
    return out;
}

}  // namespace qflab
