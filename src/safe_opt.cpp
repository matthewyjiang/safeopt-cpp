#include "safeopt/safe_opt.hpp"
#include "safeopt/utilities.hpp"
#include <algorithm>
#include <stdexcept>

namespace safeopt {

SafeOpt::SafeOpt(
    std::vector<std::shared_ptr<gp::GaussianProcess>> gps,
    const std::vector<double>& fmin,
    const std::vector<std::pair<double, double>>& bounds,
    double beta,
    const std::vector<double>& lipschitz,
    double threshold,
    const std::vector<double>& scaling)
    : GaussianProcessOptimization(gps, fmin, beta, 0, threshold, scaling),
      lipschitz_(lipschitz),
      use_lipschitz_(!lipschitz.empty()),
      best_lower_bound_(-std::numeric_limits<double>::infinity()) {
    
    setBounds(bounds);
    
    if (use_lipschitz_ && lipschitz_.size() != gps.size()) {
        throw std::invalid_argument("Lipschitz constants size must match number of GPs");
    }
    
    initialize();
}

SafeOpt::SafeOpt(
    std::vector<std::shared_ptr<gp::GaussianProcess>> gps,
    const std::vector<double>& fmin,
    const std::vector<std::pair<double, double>>& bounds,
    std::function<double(int)> beta_func,
    const std::vector<double>& lipschitz,
    double threshold,
    const std::vector<double>& scaling)
    : GaussianProcessOptimization(gps, fmin, beta_func, 0, threshold, scaling),
      lipschitz_(lipschitz),
      use_lipschitz_(!lipschitz.empty()),
      best_lower_bound_(-std::numeric_limits<double>::infinity()) {
    
    setBounds(bounds);
    
    if (use_lipschitz_ && lipschitz_.size() != gps.size()) {
        throw std::invalid_argument("Lipschitz constants size must match number of GPs");
    }
    
    initialize();
}

void SafeOpt::setInputs(const Eigen::MatrixXd& inputs) {
    inputs_ = inputs;
    int n_inputs = inputs.rows();
    int n_gps = gps_.size();
    
    // Initialize sets
    S_ = Eigen::VectorXi::Zero(n_inputs);
    M_ = Eigen::VectorXi::Zero(n_inputs);
    G_ = Eigen::VectorXi::Zero(n_inputs);
    
    // Initialize confidence intervals matrix
    Q_ = Eigen::MatrixXd::Zero(n_inputs, 2 * n_gps);
    
    computeSets(false);
}

Eigen::VectorXd SafeOpt::getMaximum(const Eigen::VectorXd& context) {
    if (context.size() > 0) {
        throw std::invalid_argument("Contextual optimization not yet supported");
    }
    
    // Compute current sets
    computeSets(false);
    
    // First try to find a maximizer
    for (int i = 0; i < inputs_.rows(); ++i) {
        if (M_[i]) {
            return inputs_.row(i);
        }
    }
    
    // If no maximizers, find best expander
    double best_acquisition = -std::numeric_limits<double>::infinity();
    int best_idx = -1;
    
    for (int i = 0; i < inputs_.rows(); ++i) {
        if (G_[i]) {
            double acq_val = getAcquisitionValue(i);
            if (acq_val > best_acquisition) {
                best_acquisition = acq_val;
                best_idx = i;
            }
        }
    }
    
    if (best_idx >= 0) {
        return inputs_.row(best_idx);
    }
    
    throw std::runtime_error("No safe points available for evaluation");
}

void SafeOpt::computeSets(bool full_sets, int num_expanders, const Eigen::VectorXd& goal) {
    if (inputs_.rows() == 0) {
        throw std::runtime_error("Input discretization not set. Call setInputs() first.");
    }
    
    computeConfidenceIntervals();
    updateSafeSet();
    updateMaximizers();
    updateExpanders(num_expanders, goal);
}

void SafeOpt::initialize() {
    // Simple initialization - in practice would need discretization
    if (bounds_.empty()) {
        throw std::runtime_error("Bounds must be set before initialization");
    }
    
    // Create a simple grid discretization for initialization
    std::vector<int> n_samples(bounds_.size(), 10);  // 10 points per dimension
    inputs_ = linearly_spaced_combinations(bounds_, n_samples);
    
    setInputs(inputs_);
}

void SafeOpt::computeConfidenceIntervals() {
    int n_inputs = inputs_.rows();
    double beta_val = beta_(getT());
    
    for (size_t gp_idx = 0; gp_idx < gps_.size(); ++gp_idx) {
        auto [mean, var] = gps_[gp_idx]->predict(inputs_);
        
        Eigen::VectorXd std_dev = var.array().sqrt();
        Eigen::VectorXd confidence = beta_val * std_dev / scaling_[gp_idx];
        
        // Store lower and upper bounds
        Q_.col(2 * gp_idx) = mean - confidence;      // Lower bounds
        Q_.col(2 * gp_idx + 1) = mean + confidence;  // Upper bounds
    }
}

void SafeOpt::updateSafeSet() {
    int n_inputs = inputs_.rows();
    
    for (int i = 0; i < n_inputs; ++i) {
        bool is_safe = true;
        
        // Check all constraint GPs
        for (size_t gp_idx = 0; gp_idx < gps_.size(); ++gp_idx) {
            if (fmin_[gp_idx] != -std::numeric_limits<double>::infinity()) {
                double lower_bound = Q_(i, 2 * gp_idx);
                if (lower_bound < fmin_[gp_idx]) {
                    is_safe = false;
                    break;
                }
            }
        }
        
        S_[i] = is_safe ? 1 : 0;
    }
}

void SafeOpt::updateMaximizers() {
    M_.setZero();
    best_lower_bound_ = -std::numeric_limits<double>::infinity();
    
    // Find best lower bound among safe points
    for (int i = 0; i < inputs_.rows(); ++i) {
        if (S_[i]) {
            double lower_bound = Q_(i, 0);  // Lower bound of objective GP
            if (lower_bound > best_lower_bound_) {
                best_lower_bound_ = lower_bound;
            }
        }
    }
    
    // Mark maximizers
    for (int i = 0; i < inputs_.rows(); ++i) {
        if (S_[i]) {
            double upper_bound = Q_(i, 1);  // Upper bound of objective GP
            if (upper_bound >= best_lower_bound_) {
                M_[i] = 1;
            }
        }
    }
}

void SafeOpt::updateExpanders(int num_expanders, const Eigen::VectorXd& goal) {
    G_.setZero();
    
    // Simple expander selection - points with high uncertainty
    std::vector<std::pair<double, int>> candidates;
    
    for (int i = 0; i < inputs_.rows(); ++i) {
        if (S_[i] && !M_[i]) {  // Safe but not maximizer
            double uncertainty = Q_(i, 1) - Q_(i, 0);  // Upper - lower for objective
            candidates.push_back({uncertainty, i});
        }
    }
    
    // Sort by uncertainty (descending)
    std::sort(candidates.begin(), candidates.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Select top expanders
    int count = std::min(num_expanders, static_cast<int>(candidates.size()));
    for (int i = 0; i < count; ++i) {
        G_[candidates[i].second] = 1;
    }
}

bool SafeOpt::canExpandSafeSet(int point_idx, int gp_idx) const {
    // Simplified implementation - would use Lipschitz bounds in full version
    return S_[point_idx];
}

double SafeOpt::getAcquisitionValue(int point_idx) const {
    // Simple acquisition function - uncertainty in objective
    return Q_(point_idx, 1) - Q_(point_idx, 0);
}

} // namespace safeopt