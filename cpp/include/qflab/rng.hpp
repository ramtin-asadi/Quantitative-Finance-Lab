#pragma once

#include <cmath>
#include <random>

namespace qflab {

inline double standard_normal(std::mt19937& rng) {
    static thread_local bool has_spare = false;
    static thread_local double spare = 0.0;
    if (has_spare) {
        has_spare = false;
        return spare;
    }
    std::uniform_real_distribution<double> unif(1e-12, 1.0);
    const double u1 = unif(rng);
    const double u2 = unif(rng);
    const double radius = std::sqrt(-2.0 * std::log(u1));
    const double angle = 6.2831853071795864769 * u2;
    spare = radius * std::sin(angle);
    has_spare = true;
    return radius * std::cos(angle);
}

}  // namespace qflab
