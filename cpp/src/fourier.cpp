#include "qflab/fourier.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace qflab {
namespace {

using cd = std::complex<double>;
constexpr double pi = 3.141592653589793238462643383279502884;

cd cf_bsm(cd u, const std::vector<double>& p, double s, double r, double q, double tau) {
    const double sigma = p.size() > 0 ? p[0] : 0.20;
    const double mu = std::log(s) + (r - q - 0.5 * sigma * sigma) * tau;
    return std::exp(cd(0.0, 1.0) * u * mu - 0.5 * sigma * sigma * u * u * tau);
}

cd cf_merton(cd u, const std::vector<double>& p, double s, double r, double q, double tau) {
    const double sigma = p.size() > 0 ? p[0] : 0.20;
    const double lam = p.size() > 1 ? p[1] : 0.30;
    const double muj = p.size() > 2 ? p[2] : -0.05;
    const double sj = p.size() > 3 ? p[3] : 0.20;
    const double omega = -lam * (std::exp(muj + 0.5 * sj * sj) - 1.0);
    const cd iu = cd(0.0, 1.0) * u;
    return std::exp(iu * (std::log(s) + (r - q + omega - 0.5 * sigma * sigma) * tau)
        - 0.5 * sigma * sigma * u * u * tau
        + lam * tau * (std::exp(iu * muj - 0.5 * sj * sj * u * u) - cd(1.0, 0.0)));
}

cd cf_vg(cd u, const std::vector<double>& p, double s, double r, double q, double tau) {
    const double sigma = p.size() > 0 ? p[0] : 0.20;
    const double theta = p.size() > 1 ? p[1] : -0.08;
    const double nu = p.size() > 2 ? std::max(p[2], 1e-6) : 0.20;
    const double mart = 1.0 - theta * nu - 0.5 * sigma * sigma * nu;
    const double omega = std::log(std::max(mart, 1e-8)) / nu;
    const cd iu = cd(0.0, 1.0) * u;
    return std::exp(iu * (std::log(s) + (r - q + omega) * tau))
        * std::pow(cd(1.0, 0.0) - iu * theta * nu + 0.5 * sigma * sigma * nu * u * u, -tau / nu);
}

cd cf_heston_base(cd u, const std::vector<double>& p, double s, double r, double q, double tau, double drift_adj) {
    const double v0 = p.size() > 0 ? std::max(p[0], 1e-8) : 0.04;
    const double kappa = p.size() > 1 ? std::max(p[1], 1e-8) : 2.0;
    const double theta = p.size() > 2 ? std::max(p[2], 1e-8) : 0.04;
    const double sigv = p.size() > 3 ? std::max(p[3], 1e-8) : 0.60;
    const double rho = p.size() > 4 ? std::clamp(p[4], -0.999, 0.999) : -0.50;
    const cd i(0.0, 1.0);
    const cd d = std::sqrt(std::pow(rho * sigv * i * u - kappa, 2.0) + sigv * sigv * (i * u + u * u));
    const cd g = (kappa - rho * sigv * i * u - d) / (kappa - rho * sigv * i * u + d);
    const cd expdt = std::exp(-d * tau);
    const cd c = (r - q + drift_adj) * i * u * tau
        + (kappa * theta / (sigv * sigv))
            * ((kappa - rho * sigv * i * u - d) * tau - cd(2.0, 0.0) * std::log((cd(1.0, 0.0) - g * expdt) / (cd(1.0, 0.0) - g)));
    const cd dcoef = ((kappa - rho * sigv * i * u - d) / (sigv * sigv)) * ((cd(1.0, 0.0) - expdt) / (cd(1.0, 0.0) - g * expdt));
    return std::exp(i * u * std::log(s) + c + dcoef * v0);
}

cd cf_heston(cd u, const std::vector<double>& p, double s, double r, double q, double tau) {
    return cf_heston_base(u, p, s, r, q, tau, 0.0);
}

cd cf_bates(cd u, const std::vector<double>& p, double s, double r, double q, double tau) {
    std::vector<double> hp(5, 0.0);
    for (int i = 0; i < 5 && i < static_cast<int>(p.size()); ++i) {
        hp[i] = p[i];
    }
    const double lam = p.size() > 5 ? p[5] : 0.30;
    const double muj = p.size() > 6 ? p[6] : -0.05;
    const double sj = p.size() > 7 ? p[7] : 0.20;
    const double omega = -lam * (std::exp(muj + 0.5 * sj * sj) - 1.0);
    const cd iu = cd(0.0, 1.0) * u;
    const cd jump = std::exp(lam * tau * (std::exp(iu * muj - 0.5 * sj * sj * u * u) - cd(1.0, 0.0)));
    return cf_heston_base(u, hp, s, r, q, tau, omega) * jump;
}

cd cf_model(int model_id, cd u, const std::vector<double>& p, double s, double r, double q, double tau) {
    if (model_id == 1) {
        return cf_merton(u, p, s, r, q, tau);
    }
    if (model_id == 2) {
        return cf_vg(u, p, s, r, q, tau);
    }
    if (model_id == 3) {
        return cf_heston(u, p, s, r, q, tau);
    }
    if (model_id == 4) {
        return cf_bates(u, p, s, r, q, tau);
    }
    return cf_bsm(u, p, s, r, q, tau);
}

double variance_rate(int model_id, const std::vector<double>& p, double tau) {
    if (model_id == 1) {
        const double sigma = p.size() > 0 ? p[0] : 0.20;
        const double lam = p.size() > 1 ? std::max(p[1], 0.0) : 0.30;
        const double muj = p.size() > 2 ? p[2] : -0.05;
        const double sj = p.size() > 3 ? std::max(p[3], 1e-10) : 0.20;
        return sigma * sigma + lam * (muj * muj + sj * sj);
    }
    if (model_id == 2) {
        const double sigma = p.size() > 0 ? p[0] : 0.20;
        const double theta = p.size() > 1 ? p[1] : -0.08;
        const double nu = p.size() > 2 ? std::max(p[2], 1e-10) : 0.20;
        return sigma * sigma + theta * theta * nu;
    }
    if (model_id == 3 || model_id == 4) {
        const double v0 = p.size() > 0 ? std::max(p[0], 1e-10) : 0.04;
        const double kappa = p.size() > 1 ? std::max(p[1], 1e-10) : 2.0;
        const double theta = p.size() > 2 ? std::max(p[2], 1e-10) : 0.04;
        double base = theta + (v0 - theta) * (1.0 - std::exp(-kappa * tau)) / std::max(kappa * tau, 1e-10);
        if (model_id == 4) {
            const double lam = p.size() > 5 ? std::max(p[5], 0.0) : 0.30;
            const double muj = p.size() > 6 ? p[6] : -0.05;
            const double sj = p.size() > 7 ? std::max(p[7], 1e-10) : 0.20;
            base += lam * (muj * muj + sj * sj);
        }
        return base;
    }
    const double sigma = p.size() > 0 ? p[0] : 0.20;
    return sigma * sigma;
}

std::pair<double, double> chi_psi(int n, double a, double b, double c, double d) {
    if (d <= c) {
        return {0.0, 0.0};
    }
    const double width = b - a;
    if (n == 0) {
        return {std::exp(d) - std::exp(c), d - c};
    }
    const double u = static_cast<double>(n) * pi / width;
    const double xd = u * (d - a);
    const double xc = u * (c - a);
    const double chi = (
        std::cos(xd) * std::exp(d)
        - std::cos(xc) * std::exp(c)
        + u * (std::sin(xd) * std::exp(d) - std::sin(xc) * std::exp(c))
    ) / (1.0 + u * u);
    const double psi = (std::sin(xd) - std::sin(xc)) / u;
    return {chi, psi};
}

double cos_one(int model_id, const std::vector<double>& p, double s, double k, double r, double q, double tau, int n_terms, double width_mult, int option_type) {
    const double var = std::max(variance_rate(model_id, p, tau), 1e-10);
    const double drift = std::log(s / k) + (r - q) * tau;
    const double stdev = std::sqrt(std::max(var * tau, 1e-10));
    const double half = std::max(width_mult * stdev + 0.10 * std::abs(drift), 0.35);
    const double a = drift - half;
    const double b = drift + half;
    const double width = b - a;
    double total = 0.0;
    const cd i(0.0, 1.0);
    for (int n = 0; n < std::max(8, n_terms); ++n) {
        const double u = static_cast<double>(n) * pi / width;
        const cd phi = cf_model(model_id, cd(u, 0.0), p, s, r, q, tau) * std::exp(-i * u * std::log(k));
        const cd phase = std::exp(-i * u * a);
        double coeff = 0.0;
        if (option_type > 0) {
            const auto cp = chi_psi(n, a, b, std::max(0.0, a), b);
            coeff = 2.0 * k * (cp.first - cp.second) / width;
        } else {
            const auto cp = chi_psi(n, a, b, a, std::min(0.0, b));
            coeff = 2.0 * k * (cp.second - cp.first) / width;
        }
        double term = std::real(phi * phase) * coeff;
        if (n == 0) {
            term *= 0.5;
        }
        total += term;
    }
    return std::max(std::exp(-r * tau) * total, 0.0);
}

double direct_call(int model_id, const std::vector<double>& p, double s, double k, double r, double q, double tau, int n_terms, double u_max) {
    const cd i(0.0, 1.0);
    const double logk = std::log(k);
    const int n = std::max(64, n_terms);
    const double umax = std::max(10.0, u_max);
    const double du = umax / static_cast<double>(n);
    const cd phi_minus_i = cf_model(model_id, cd(0.0, -1.0), p, s, r, q, tau);
    double p1 = 0.5;
    double p2 = 0.5;
    for (int j = 1; j <= n; ++j) {
        const double u = (j - 0.5) * du;
        const cd uc(u, 0.0);
        const cd e = std::exp(-i * uc * logk);
        const cd den = i * uc;
        const cd a1 = e * cf_model(model_id, uc - i, p, s, r, q, tau) / (den * phi_minus_i);
        const cd a2 = e * cf_model(model_id, uc, p, s, r, q, tau) / den;
        p1 += du / pi * std::real(a1);
        p2 += du / pi * std::real(a2);
    }
    const double call = s * std::exp(-q * tau) * p1 - k * std::exp(-r * tau) * p2;
    return std::max(call, 0.0);
}

std::vector<double> params_vec(py::array_t<double, py::array::c_style | py::array::forcecast> params) {
    const auto p = params.unchecked<1>();
    std::vector<double> out(static_cast<size_t>(p.shape(0)));
    for (py::ssize_t idx = 0; idx < p.shape(0); ++idx) {
        out[static_cast<size_t>(idx)] = p(idx);
    }
    return out;
}

int next_power_two(int n) {
    int out = 1;
    while (out < n) {
        out <<= 1;
    }
    return out;
}

void fft_inplace(std::vector<cd>& x) {
    const int n = static_cast<int>(x.size());
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(x[static_cast<size_t>(i)], x[static_cast<size_t>(j)]);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const double angle = -2.0 * pi / static_cast<double>(len);
        const cd wlen(std::cos(angle), std::sin(angle));
        const int half = len >> 1;
        for (int i = 0; i < n; i += len) {
            cd w(1.0, 0.0);
            for (int k = 0; k < half; ++k) {
                const cd u = x[static_cast<size_t>(i + k)];
                const cd v = x[static_cast<size_t>(i + k + half)] * w;
                x[static_cast<size_t>(i + k)] = u + v;
                x[static_cast<size_t>(i + k + half)] = u - v;
                w *= wlen;
            }
        }
    }
}

}  // namespace

py::dict fft_prices(
    int model_id,
    py::array_t<double, py::array::c_style | py::array::forcecast> params,
    double s,
    double r,
    double q,
    double tau,
    double alpha,
    int n,
    double eta,
    int option_type) {
    const std::vector<double> p = params_vec(params);
    const int nn = next_power_two(std::max(16, n));
    const double lambda = 2.0 * pi / (static_cast<double>(nn) * eta);
    const double center = std::log(s);
    const double start = center - 0.5 * static_cast<double>(nn) * lambda;
    auto strikes = py::array_t<double>(nn);
    auto prices = py::array_t<double>(nn);
    auto kk = strikes.mutable_unchecked<1>();
    auto px = prices.mutable_unchecked<1>();
    const cd i(0.0, 1.0);
    std::vector<cd> x(static_cast<size_t>(nn));
    for (int j = 0; j < nn; ++j) {
        const double u = static_cast<double>(j) * eta;
        const cd uc(u, 0.0);
        const cd shifted = uc - i * (alpha + 1.0);
        const cd denom = alpha * alpha + alpha - u * u + i * (2.0 * alpha + 1.0) * u;
        const double weight = j == 0 ? 0.5 : 1.0;
        const cd psi = std::exp(-r * tau) * cf_model(model_id, shifted, p, s, r, q, tau) / denom;
        x[static_cast<size_t>(j)] = psi * std::exp(-i * uc * start) * eta * weight;
    }
    fft_inplace(x);
    for (int m = 0; m < nn; ++m) {
        const double logk = start + lambda * static_cast<double>(m);
        const double k = std::exp(logk);
        kk(m) = k;
        double call = std::exp(-alpha * logk) * std::real(x[static_cast<size_t>(m)]) / pi;
        call = std::max(call, 0.0);
        if (option_type > 0) {
            px(m) = call;
        } else {
            px(m) = call - s * std::exp(-q * tau) + k * std::exp(-r * tau);
        }
    }
    py::dict out;
    out["strikes"] = strikes;
    out["prices"] = prices;
    return out;
}

py::array_t<double> direct_prices(
    int model_id,
    py::array_t<double, py::array::c_style | py::array::forcecast> params,
    py::array_t<double, py::array::c_style | py::array::forcecast> strikes,
    py::array_t<double, py::array::c_style | py::array::forcecast> tau,
    double s,
    double r,
    double q,
    int n_terms,
    double u_max,
    int option_type) {
    const auto ks = strikes.unchecked<1>();
    const auto tt = tau.unchecked<1>();
    if (tt.shape(0) != ks.shape(0)) {
        throw std::runtime_error("strikes and tau must have the same length");
    }
    const std::vector<double> p = params_vec(params);
    auto out = py::array_t<double>(ks.shape(0));
    auto y = out.mutable_unchecked<1>();
    if (ks.shape(0) == 0) {
        return out;
    }
    bool same_tau = true;
    const double tau0 = tt(0);
    for (py::ssize_t idx = 1; idx < tt.shape(0); ++idx) {
        if (std::abs(tt(idx) - tau0) > 1e-13) {
            same_tau = false;
            break;
        }
    }
    const int n = std::max(64, n_terms);
    const double umax = std::max(10.0, u_max);
    if (same_tau) {
        const cd i(0.0, 1.0);
        const double du = umax / static_cast<double>(n);
        const cd phi_minus_i = cf_model(model_id, cd(0.0, -1.0), p, s, r, q, tau0);
        std::vector<double> p1(static_cast<size_t>(ks.shape(0)), 0.5);
        std::vector<double> p2(static_cast<size_t>(ks.shape(0)), 0.5);
        for (int j = 1; j <= n; ++j) {
            const double u = (j - 0.5) * du;
            const cd uc(u, 0.0);
            const cd den = i * uc;
            const cd a1 = cf_model(model_id, uc - i, p, s, r, q, tau0) / (den * phi_minus_i);
            const cd a2 = cf_model(model_id, uc, p, s, r, q, tau0) / den;
            const double weight = du / pi;
            for (py::ssize_t idx = 0; idx < ks.shape(0); ++idx) {
                const cd e = std::exp(-i * uc * std::log(ks(idx)));
                p1[static_cast<size_t>(idx)] += weight * std::real(e * a1);
                p2[static_cast<size_t>(idx)] += weight * std::real(e * a2);
            }
        }
        const double stock_disc = s * std::exp(-q * tau0);
        const double bond_disc = std::exp(-r * tau0);
        for (py::ssize_t idx = 0; idx < ks.shape(0); ++idx) {
            double call = stock_disc * p1[static_cast<size_t>(idx)] - ks(idx) * bond_disc * p2[static_cast<size_t>(idx)];
            call = std::max(call, 0.0);
            if (option_type > 0) {
                y(idx) = call;
            } else {
                y(idx) = call - stock_disc + ks(idx) * bond_disc;
            }
        }
        return out;
    }
    for (py::ssize_t idx = 0; idx < ks.shape(0); ++idx) {
        const double call = direct_call(model_id, p, s, ks(idx), r, q, tt(idx), n, umax);
        if (option_type > 0) {
            y(idx) = call;
        } else {
            y(idx) = call - s * std::exp(-q * tt(idx)) + ks(idx) * std::exp(-r * tt(idx));
        }
    }
    return out;
}

py::array_t<double> cos_prices(
    int model_id,
    py::array_t<double, py::array::c_style | py::array::forcecast> params,
    py::array_t<double, py::array::c_style | py::array::forcecast> strikes,
    py::array_t<double, py::array::c_style | py::array::forcecast> tau,
    double s,
    double r,
    double q,
    int n_terms,
    double truncation_width,
    int option_type) {
    const auto ks = strikes.unchecked<1>();
    const auto tt = tau.unchecked<1>();
    if (tt.shape(0) != ks.shape(0)) {
        throw std::runtime_error("strikes and tau must have the same length");
    }
    const std::vector<double> p = params_vec(params);
    auto out = py::array_t<double>(ks.shape(0));
    auto y = out.mutable_unchecked<1>();
    for (py::ssize_t idx = 0; idx < ks.shape(0); ++idx) {
        y(idx) = cos_one(model_id, p, s, ks(idx), r, q, tt(idx), n_terms, truncation_width, option_type);
    }
    return out;
}

}  // namespace qflab
