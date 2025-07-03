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
    
    // Check if we have any safe points
    bool has_safe_points = false;
    for (int i = 0; i < inputs_.rows(); ++i) {
        if (S_[i]) {
            has_safe_points = true;
            break;
        }
    }
    
    if (!has_safe_points) {
        throw std::runtime_error("No safe points available for evaluation");
    }
    
    // Find the best point among maximizers and expanders
    double best_acquisition = -std::numeric_limits<double>::infinity();
    int best_idx = -1;
    
    // Check maximizers and expanders (union of M and G)
    for (int i = 0; i < inputs_.rows(); ++i) {
        if (M_[i] || G_[i]) {
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
    
    // Fallback: return any safe point if no maximizers or expanders
    for (int i = 0; i < inputs_.rows(); ++i) {
        if (S_[i]) {
            return inputs_.row(i);
        }
    }
    
    throw std::runtime_error("No safe points available for evaluation");
}

Eigen::VectorXd SafeOpt::optimize(const Eigen::VectorXd& context, bool ucb_only) {
    if (context.size() > 0) {
        throw std::invalid_argument("Contextual optimization not yet supported");
    }
    
    // Update confidence intervals and compute sets
    computeConfidenceIntervals();
    
    if (ucb_only) {
        // Only compute safe set, skip expanders
        updateSafeSet();
        updateMaximizers();
    } else {
        // Compute full sets including expanders
        computeSets(true);
    }
    
    // Check if we have any safe points
    bool has_safe_points = false;
    for (int i = 0; i < inputs_.rows(); ++i) {
        if (S_[i]) {
            has_safe_points = true;
            break;
        }
    }
    
    if (!has_safe_points) {
        throw std::runtime_error("No safe points available for evaluation");
    }
    
    if (ucb_only) {
        // Use Safe-UCB criterion: find safe point with highest UCB
        double best_ucb = -std::numeric_limits<double>::infinity();
        int best_idx = -1;
        
        for (int i = 0; i < inputs_.rows(); ++i) {
            if (S_[i]) {
                double ucb = getUCB(i, 0);  // UCB of objective function
                if (ucb > best_ucb) {
                    best_ucb = ucb;
                    best_idx = i;
                }
            }
        }
        
        if (best_idx >= 0) {
            return inputs_.row(best_idx);
        }
    } else {
        // Use full SafeOpt criterion: choose from maximizers and expanders
        return getMaximum(context);
    }
    
    throw std::runtime_error("No suitable points found for evaluation");
}

void SafeOpt::computeSets(bool full_sets, int num_expanders, const Eigen::VectorXd& goal) {
    if (inputs_.rows() == 0) {
        throw std::runtime_error("Input discretization not set. Call setInputs() first.");
    }
    
    computeConfidenceIntervals();
    updateSafeSet();
    updateMaximizers();
    updateExpanders(full_sets ? num_expanders : 1, goal);  // Use full_sets parameter
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
    
    double beta_val = beta_(getT());
    
    // Find maximum variance in maximizers to filter candidates
    double max_var = 0.0;
    for (int i = 0; i < inputs_.rows(); ++i) {
        if (M_[i]) {
            double var = (Q_(i, 1) - Q_(i, 0)) / scaling_[0];
            max_var = std::max(max_var, var);
        }
    }
    
    // Collect safe candidate points (excluding maximizers)
    std::vector<int> safe_candidates;
    for (int i = 0; i < inputs_.rows(); ++i) {
        if (S_[i] && !M_[i]) {
            // Check if variance is high enough
            double max_uncertainty = 0.0;
            for (size_t gp_idx = 0; gp_idx < gps_.size(); ++gp_idx) {
                double uncertainty = (Q_(i, 2*gp_idx + 1) - Q_(i, 2*gp_idx)) / scaling_[gp_idx];
                max_uncertainty = std::max(max_uncertainty, uncertainty);
            }
            
            if (max_uncertainty > max_var) {
                // Check threshold condition
                bool above_threshold = false;
                for (size_t gp_idx = 0; gp_idx < gps_.size(); ++gp_idx) {
                    double uncertainty = Q_(i, 2*gp_idx + 1) - Q_(i, 2*gp_idx);
                    if (uncertainty > threshold_ * beta_val) {
                        above_threshold = true;
                        break;
                    }
                }
                
                if (above_threshold) {
                    safe_candidates.push_back(i);
                }
            }
        }
    }
    
    if (safe_candidates.empty()) {
        return;
    }
    
    // Sort candidates by uncertainty (descending) or distance to goal
    std::vector<std::pair<double, int>> scored_candidates;
    
    for (int idx : safe_candidates) {
        double score;
        if (goal.size() > 0) {
            // Sort by distance to goal (smaller distance = higher score)
            Eigen::VectorXd diff = inputs_.row(idx) - goal.transpose();
            score = -diff.norm();
        } else {
            // Sort by maximum uncertainty across all GPs
            score = 0.0;
            for (size_t gp_idx = 0; gp_idx < gps_.size(); ++gp_idx) {
                double uncertainty = Q_(idx, 2*gp_idx + 1) - Q_(idx, 2*gp_idx);
                score = std::max(score, uncertainty);
            }
        }
        scored_candidates.push_back({score, idx});
    }
    
    // Sort by score (descending)
    std::sort(scored_candidates.begin(), scored_candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Check each candidate to see if it's an expander
    int expanders_found = 0;
    for (const auto& candidate : scored_candidates) {
        int idx = candidate.second;
        
        if (canExpandSafeSet(idx)) {
            G_[idx] = 1;
            expanders_found++;
            
            if (expanders_found >= num_expanders) {
                break;
            }
        }
    }
}

bool SafeOpt::canExpandSafeSet(int point_idx, int gp_idx) const {
    // Skip if no safety constraint for this GP
    if (fmin_[gp_idx] == -std::numeric_limits<double>::infinity()) {
        return true;
    }
    
    double beta_val = beta_(getT());
    
    if (use_lipschitz_) {
        // Lipschitz-based expansion check
        // Find unsafe points
        std::vector<int> unsafe_points;
        for (int i = 0; i < inputs_.rows(); ++i) {
            if (!S_[i]) {
                unsafe_points.push_back(i);
            }
        }
        
        if (unsafe_points.empty()) {
            return true;  // No unsafe points to expand to
        }
        
        // Check if this point can make any unsafe point safe
        Eigen::VectorXd point = inputs_.row(point_idx);
        double upper_bound = Q_(point_idx, 2 * gp_idx + 1);
        
        for (int unsafe_idx : unsafe_points) {
            Eigen::VectorXd unsafe_point = inputs_.row(unsafe_idx);
            double distance = (point - unsafe_point).norm();
            
            // Safety check: u - L * d >= fmin
            if (upper_bound - lipschitz_[gp_idx] * distance >= fmin_[gp_idx]) {
                return true;
            }
        }
        
        return false;
    } else {
        // GP-based expansion check
        // Temporarily add this point with its upper bound to the GP
        Eigen::VectorXd point = inputs_.row(point_idx).transpose();
        double upper_bound = Q_(point_idx, 2 * gp_idx + 1);
        
        // Get current GP training data
        auto current_X = gps_[gp_idx]->getX();
        auto current_y = gps_[gp_idx]->getY();
        
        // Add temporary point
        Eigen::MatrixXd temp_X(current_X.rows() + 1, current_X.cols());
        temp_X.topRows(current_X.rows()) = current_X;
        temp_X.bottomRows(1) = point.transpose();
        
        Eigen::VectorXd temp_y(current_y.size() + 1);
        temp_y.head(current_y.size()) = current_y;
        temp_y(temp_y.size() - 1) = upper_bound;
        
        // Temporarily update GP
        gps_[gp_idx]->fit(temp_X, temp_y);
        
        // Check if any previously unsafe points become safe
        bool can_expand = false;
        for (int i = 0; i < inputs_.rows(); ++i) {
            if (!S_[i]) {
                Eigen::MatrixXd test_point = inputs_.row(i);
                auto [mean, var] = gps_[gp_idx]->predict(test_point);
                double lower_bound = mean(0) - beta_val * std::sqrt(var(0));
                
                if (lower_bound >= fmin_[gp_idx]) {
                    can_expand = true;
                    break;
                }
            }
        }
        
        // Restore original GP
        gps_[gp_idx]->fit(current_X, current_y);
        
        return can_expand;
    }
}

bool SafeOpt::canExpandSafeSet(int point_idx) const {
    // Check if point can expand safe set for all constraint GPs
    for (size_t gp_idx = 0; gp_idx < gps_.size(); ++gp_idx) {
        if (fmin_[gp_idx] != -std::numeric_limits<double>::infinity()) {
            if (canExpandSafeSet(point_idx, gp_idx)) {
                return true;  // If it can expand for any GP, it's an expander
            }
        }
    }
    return false;
}

double SafeOpt::getAcquisitionValue(int point_idx) const {
    // SafeOpt acquisition function: maximum scaled uncertainty across all GPs
    double max_scaled_uncertainty = 0.0;
    
    for (size_t gp_idx = 0; gp_idx < gps_.size(); ++gp_idx) {
        double uncertainty = Q_(point_idx, 2*gp_idx + 1) - Q_(point_idx, 2*gp_idx);
        double scaled_uncertainty = uncertainty / scaling_[gp_idx];
        max_scaled_uncertainty = std::max(max_scaled_uncertainty, scaled_uncertainty);
    }
    
    return max_scaled_uncertainty;
}

double SafeOpt::getUCB(int point_idx, int gp_idx) const {
    if (gp_idx >= static_cast<int>(gps_.size())) {
        throw std::out_of_range("GP index out of range");
    }
    return Q_(point_idx, 2*gp_idx + 1);  // Upper confidence bound
}

double SafeOpt::getLCB(int point_idx, int gp_idx) const {
    if (gp_idx >= static_cast<int>(gps_.size())) {
        throw std::out_of_range("GP index out of range");
    }
    return Q_(point_idx, 2*gp_idx);  // Lower confidence bound
}

double SafeOpt::getSafeUCB(int point_idx) const {
    // Return UCB only if point is safe, otherwise return -infinity
    if (S_[point_idx]) {
        return getUCB(point_idx, 0);  // UCB of objective function
    } else {
        return -std::numeric_limits<double>::infinity();
    }
}

} // namespace safeopt