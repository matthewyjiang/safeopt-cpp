#include "safeopt/safe_opt_swarm.hpp"
#include "safeopt/utilities.hpp"
#include <algorithm>
#include <stdexcept>
#include <random>

namespace safeopt {

SafeOptSwarm::SafeOptSwarm(
    std::vector<std::shared_ptr<gp::GaussianProcess>> gps,
    const std::vector<double>& fmin,
    const std::vector<std::pair<double, double>>& bounds,
    double beta,
    double threshold,
    const std::vector<double>& scaling,
    int swarm_size,
    int max_iters)
    : GaussianProcessOptimization(gps, fmin, beta, 0, threshold, scaling),
      swarm_size_(swarm_size),
      max_iters_(max_iters),
      best_lower_bound_(-std::numeric_limits<double>::infinity()) {
    
    setBounds(bounds);
    initializeSwarms();
}

SafeOptSwarm::SafeOptSwarm(
    std::vector<std::shared_ptr<gp::GaussianProcess>> gps,
    const std::vector<double>& fmin,
    const std::vector<std::pair<double, double>>& bounds,
    std::function<double(int)> beta_func,
    double threshold,
    const std::vector<double>& scaling,
    int swarm_size,
    int max_iters)
    : GaussianProcessOptimization(gps, fmin, beta_func, 0, threshold, scaling),
      swarm_size_(swarm_size),
      max_iters_(max_iters),
      best_lower_bound_(-std::numeric_limits<double>::infinity()) {
    
    setBounds(bounds);
    initializeSwarms();
}

Eigen::VectorXd SafeOptSwarm::getMaximum(const Eigen::VectorXd& context) {
    if (context.size() > 0) {
        throw std::invalid_argument("Contextual optimization not supported in SafeOptSwarm");
    }
    
    return selectNextPoint();
}

void SafeOptSwarm::setSwarmParameters(int swarm_size, int max_iters) {
    swarm_size_ = swarm_size;
    max_iters_ = max_iters;
    initializeSwarms();
}

void SafeOptSwarm::initializeSwarms() {
    if (bounds_.empty()) {
        throw std::runtime_error("Bounds must be set before initializing swarms");
    }
    
    int ndim = bounds_.size();
    Eigen::VectorXd velocity_scale = Eigen::VectorXd::Ones(ndim);
    
    // Scale velocities based on bounds
    for (int i = 0; i < ndim; ++i) {
        velocity_scale[i] = (bounds_[i].second - bounds_[i].first) * 0.1;
    }
    
    // Create fitness functions
    auto maximizer_fitness = [this](const Eigen::MatrixXd& particles) {
        return computeMaximizerFitness(particles);
    };
    
    auto expander_fitness = [this](const Eigen::MatrixXd& particles) {
        return computeExpanderFitness(particles);
    };
    
    // Initialize swarms
    maximizer_swarm_ = std::make_unique<SwarmOptimization>(
        swarm_size_, velocity_scale, maximizer_fitness, bounds_);
        
    expander_swarm_ = std::make_unique<SwarmOptimization>(
        swarm_size_, velocity_scale, expander_fitness, bounds_);
}

std::pair<Eigen::VectorXd, Eigen::VectorXi> SafeOptSwarm::computeMaximizerFitness(
    const Eigen::MatrixXd& particles) {
    
    int n_particles = particles.rows();
    Eigen::VectorXd fitness(n_particles);
    Eigen::VectorXi safety = checkSafety(particles);
    
    // Get confidence intervals
    Eigen::MatrixXd intervals = getConfidenceIntervals(particles);
    
    for (int i = 0; i < n_particles; ++i) {
        if (safety[i]) {
            // For maximizers, fitness is upper bound of objective
            fitness[i] = intervals(i, 1);  // Upper bound of first GP (objective)
        } else {
            fitness[i] = -std::numeric_limits<double>::infinity();
        }
    }
    
    return {fitness, safety};
}

std::pair<Eigen::VectorXd, Eigen::VectorXi> SafeOptSwarm::computeExpanderFitness(
    const Eigen::MatrixXd& particles) {
    
    int n_particles = particles.rows();
    Eigen::VectorXd fitness(n_particles);
    Eigen::VectorXi safety = checkSafety(particles);
    
    // Get confidence intervals
    Eigen::MatrixXd intervals = getConfidenceIntervals(particles);
    
    for (int i = 0; i < n_particles; ++i) {
        if (safety[i]) {
            // For expanders, fitness is uncertainty (width of confidence interval)
            fitness[i] = intervals(i, 1) - intervals(i, 0);
        } else {
            fitness[i] = -std::numeric_limits<double>::infinity();
        }
    }
    
    return {fitness, safety};
}

Eigen::VectorXd SafeOptSwarm::computePenalty(const Eigen::MatrixXd& slack) const {
    // Simple penalty function - exponential penalty for constraint violations
    Eigen::VectorXd penalties = Eigen::VectorXd::Zero(slack.rows());
    
    for (int i = 0; i < slack.rows(); ++i) {
        double total_violation = 0.0;
        for (int j = 0; j < slack.cols(); ++j) {
            if (slack(i, j) < 0) {  // Constraint violated
                total_violation += std::abs(slack(i, j));
            }
        }
        penalties[i] = std::exp(total_violation) - 1.0;
    }
    
    return penalties;
}

Eigen::VectorXi SafeOptSwarm::checkSafety(const Eigen::MatrixXd& particles) const {
    int n_particles = particles.rows();
    Eigen::VectorXi safety = Eigen::VectorXi::Ones(n_particles);
    
    // Get confidence intervals for all GPs
    Eigen::MatrixXd intervals = getConfidenceIntervals(particles);
    
    for (int i = 0; i < n_particles; ++i) {
        for (size_t gp_idx = 0; gp_idx < gps_.size(); ++gp_idx) {
            if (fmin_[gp_idx] != -std::numeric_limits<double>::infinity()) {
                // Check if lower bound satisfies constraint
                double lower_bound = intervals(i, 2 * gp_idx);
                if (lower_bound < fmin_[gp_idx]) {
                    safety[i] = 0;
                    break;
                }
            }
        }
    }
    
    return safety;
}

Eigen::MatrixXd SafeOptSwarm::getConfidenceIntervals(const Eigen::MatrixXd& particles) const {
    int n_particles = particles.rows();
    int n_gps = gps_.size();
    Eigen::MatrixXd intervals(n_particles, 2 * n_gps);
    
    double beta_val = beta_(getT());
    
    for (size_t gp_idx = 0; gp_idx < gps_.size(); ++gp_idx) {
        auto [mean, var] = gps_[gp_idx]->predict(particles);
        
        Eigen::VectorXd std_dev = var.array().sqrt();
        Eigen::VectorXd confidence = beta_val * std_dev / scaling_[gp_idx];
        
        intervals.col(2 * gp_idx) = mean - confidence;      // Lower bounds
        intervals.col(2 * gp_idx + 1) = mean + confidence;  // Upper bounds
    }
    
    return intervals;
}

Eigen::MatrixXd SafeOptSwarm::sampleInitialParticles(int n_particles) const {
    Eigen::MatrixXd particles(n_particles, bounds_.size());
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int i = 0; i < n_particles; ++i) {
        for (size_t j = 0; j < bounds_.size(); ++j) {
            std::uniform_real_distribution<double> dist(bounds_[j].first, bounds_[j].second);
            particles(i, j) = dist(gen);
        }
    }
    
    return particles;
}

Eigen::VectorXd SafeOptSwarm::selectNextPoint() {
    // Try maximizer optimization first
    Eigen::VectorXd maximizer_point = optimizeMaximizers();
    
    // Try expander optimization
    Eigen::VectorXd expander_point = optimizeExpanders();
    
    // Simple selection criterion - alternate between maximizers and expanders
    // In practice, this would be more sophisticated
    static bool prefer_maximizer = true;
    prefer_maximizer = !prefer_maximizer;
    
    if (prefer_maximizer) {
        return maximizer_point;
    } else {
        return expander_point;
    }
}

Eigen::VectorXd SafeOptSwarm::optimizeMaximizers() {
    Eigen::MatrixXd initial_particles = sampleInitialParticles(swarm_size_);
    maximizer_swarm_->initSwarm(initial_particles);
    maximizer_swarm_->runSwarm(max_iters_);
    return maximizer_swarm_->getGlobalBest();
}

Eigen::VectorXd SafeOptSwarm::optimizeExpanders() {
    Eigen::MatrixXd initial_particles = sampleInitialParticles(swarm_size_);
    expander_swarm_->initSwarm(initial_particles);
    expander_swarm_->runSwarm(max_iters_);
    return expander_swarm_->getGlobalBest();
}

} // namespace safeopt