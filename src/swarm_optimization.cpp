#include "safeopt/swarm_optimization.hpp"
#include <random>
#include <algorithm>
#include <stdexcept>

namespace safeopt {

SwarmOptimization::SwarmOptimization(
    int swarm_size,
    const Eigen::VectorXd& velocity_scale,
    std::function<std::pair<Eigen::VectorXd, Eigen::VectorXi>(const Eigen::MatrixXd&)> fitness_func,
    const std::vector<std::pair<double, double>>& bounds)
    : swarm_size_(swarm_size),
      ndim_(velocity_scale.size()),
      fitness_(fitness_func),
      bounds_(bounds),
      velocity_scale_(velocity_scale) {

    if (swarm_size <= 0) {
        throw std::invalid_argument("Swarm size must be positive");
    }

    if (ndim_ <= 0) {
        throw std::invalid_argument("Velocity scale must have positive dimension");
    }

    // Initialize matrices
    positions_ = Eigen::MatrixXd::Zero(swarm_size_, ndim_);
    velocities_ = Eigen::MatrixXd::Zero(swarm_size_, ndim_);
    best_positions_ = Eigen::MatrixXd::Zero(swarm_size_, ndim_);
    best_values_ = Eigen::VectorXd::Constant(swarm_size_, -std::numeric_limits<double>::infinity());
    global_best_ = Eigen::VectorXd::Zero(ndim_);
}

void SwarmOptimization::initSwarm(const Eigen::MatrixXd& positions) {
    if (positions.rows() != swarm_size_ || positions.cols() != ndim_) {
        throw std::invalid_argument("Position matrix dimensions don't match swarm configuration");
    }

    positions_ = positions;

    // Initialize velocities randomly
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < swarm_size_; ++i) {
        for (int j = 0; j < ndim_; ++j) {
            velocities_(i, j) = dist(gen) * velocity_scale_[j];
        }
    }

    // Evaluate initial fitness
    auto [values, safe] = fitness_(positions_);

    // Initialize best estimates
    best_positions_ = positions_;
    best_values_ = values;

    // Find global best among safe particles
    double best_value = -std::numeric_limits<double>::infinity();
    int best_idx = -1;
    for (int i = 0; i < swarm_size_; ++i) {
        if (safe[i] && values[i] > best_value) {
            best_value = values[i];
            best_idx = i;
        }
    }

    if (best_idx >= 0) {
        global_best_ = best_positions_.row(best_idx);
    }
}

void SwarmOptimization::runSwarm(int max_iter) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    for (int iter = 0; iter < max_iter; ++iter) {
        // Update velocities
        updateVelocities(iter, max_iter);

        // Clip velocities
        Eigen::VectorXd max_vel = getMaxVelocity();
        for (int i = 0; i < swarm_size_; ++i) {
            for (int j = 0; j < ndim_; ++j) {
                velocities_(i, j) = std::clamp(velocities_(i, j), -max_vel[j], max_vel[j]);
            }
        }

        // Update positions
        positions_ += velocities_;

        // Clip to bounds if specified
        if (!bounds_.empty()) {
            clipToBounds(positions_, bounds_);
        }

        // Evaluate fitness
        auto [values, safe] = fitness_(positions_);

        // Update best positions
        for (int i = 0; i < swarm_size_; ++i) {
            if (safe[i] && values[i] > best_values_[i]) {
                best_values_[i] = values[i];
                best_positions_.row(i) = positions_.row(i);
            }
        }

        // Update global best
        double best_value = -std::numeric_limits<double>::infinity();
        int best_idx = -1;
        for (int i = 0; i < swarm_size_; ++i) {
            if (best_values_[i] > best_value) {
                best_value = best_values_[i];
                best_idx = i;
            }
        }

        if (best_idx >= 0) {
            global_best_ = best_positions_.row(best_idx);
        }
    }
}

void SwarmOptimization::updateVelocities(int iteration, int max_iter) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    double inertia = computeInertia(iteration, max_iter);

    for (int i = 0; i < swarm_size_; ++i) {
        for (int j = 0; j < ndim_; ++j) {
            double r1 = uniform(gen);
            double r2 = uniform(gen);

            double cognitive = c1_ * r1 * (best_positions_(i, j) - positions_(i, j));
            double social = c2_ * r2 * (global_best_[j] - positions_(i, j));

            velocities_(i, j) = inertia * velocities_(i, j) + cognitive + social;
        }
    }
}

double SwarmOptimization::computeInertia(int iteration, int max_iter) const {
    if (max_iter <= 1) {
        return initial_inertia_;
    }

    double ratio = static_cast<double>(iteration) / (max_iter - 1);
    return initial_inertia_ - (initial_inertia_ - final_inertia_) * ratio;
}

void SwarmOptimization::clipToBounds(
    Eigen::MatrixXd& values, 
    const std::vector<std::pair<double, double>>& bounds) const {
    
    if (bounds.size() != static_cast<size_t>(values.cols())) {
        throw std::invalid_argument("Bounds size must match matrix columns");
    }

    for (int i = 0; i < values.rows(); ++i) {
        for (int j = 0; j < values.cols(); ++j) {
            values(i, j) = std::clamp(values(i, j), bounds[j].first, bounds[j].second);
        }
    }
}

} // namespace safeopt