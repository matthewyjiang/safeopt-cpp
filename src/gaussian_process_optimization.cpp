#include "safeopt/gaussian_process_optimization.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace safeopt {

GaussianProcessOptimization::GaussianProcessOptimization(
    std::vector<std::shared_ptr<gp::GaussianProcess>> gps,
    const std::vector<double>& fmin,
    double beta,
    int num_contexts,
    double threshold,
    const std::vector<double>& scaling)
    : GaussianProcessOptimization(gps, fmin, 
                                 [beta](int) { return beta; },
                                 num_contexts, threshold, scaling) {
}

GaussianProcessOptimization::GaussianProcessOptimization(
    std::vector<std::shared_ptr<gp::GaussianProcess>> gps,
    const std::vector<double>& fmin,
    std::function<double(int)> beta_func,
    int num_contexts,
    double threshold,
    const std::vector<double>& scaling)
    : gps_(std::move(gps)),
      gp_(gps_.empty() ? nullptr : gps_[0]),
      fmin_(fmin),
      beta_(beta_func),
      threshold_(threshold),
      num_contexts_(num_contexts),
      num_samples_(0) {

    if (gps_.empty()) {
        throw std::invalid_argument("At least one GP must be provided");
    }

    // Ensure fmin has correct size
    if (fmin_.size() != gps_.size()) {
        if (fmin_.size() == 1) {
            // Broadcast single value
            fmin_.resize(gps_.size(), fmin_[0]);
        } else {
            throw std::invalid_argument("fmin size must match number of GPs");
        }
    }

    // Set up scaling
    if (scaling.empty()) {
        computeAutoScaling();
    } else {
        if (scaling.size() != gps_.size()) {
            throw std::invalid_argument("scaling size must match number of GPs");
        }
        scaling_ = scaling;
    }

    // Initialize data
    getInitialXY();
}

void GaussianProcessOptimization::addNewDataPoint(
    const Eigen::VectorXd& x, 
    const Eigen::VectorXd& y, 
    const Eigen::VectorXd& context) {
    
    if (y.size() != static_cast<int>(gps_.size())) {
        throw std::invalid_argument("y size must match number of GPs");
    }

    // Add context if provided
    Eigen::VectorXd x_with_context = addContext(x, context);

    // Add to each GP
    for (size_t i = 0; i < gps_.size(); ++i) {
        addDataPoint(gps_[i], x_with_context, y[i], context);
    }

    // Update internal data matrices
    int old_size = x_.rows();
    x_.conservativeResize(old_size + 1, Eigen::NoChange);
    y_.conservativeResize(old_size + 1, Eigen::NoChange);
    
    x_.row(old_size) = x_with_context;
    y_.row(old_size) = y.transpose();
    
    num_samples_++;
}

void GaussianProcessOptimization::removeLastDataPoint() {
    if (x_.rows() == 0) {
        throw std::runtime_error("No data points to remove");
    }

    // Remove from each GP
    for (auto& gp : gps_) {
        gp->removeLastDataPoint();
    }

    // Update internal data matrices
    int new_size = x_.rows() - 1;
    x_.conservativeResize(new_size, Eigen::NoChange);
    y_.conservativeResize(new_size, Eigen::NoChange);
    
    num_samples_--;
}

void GaussianProcessOptimization::setBounds(
    const std::vector<std::pair<double, double>>& bounds) {
    bounds_ = bounds;
}

void GaussianProcessOptimization::getInitialXY() {
    if (!gp_ || gp_->getX().rows() == 0) {
        // No initial data
        x_ = Eigen::MatrixXd(0, 0);
        y_ = Eigen::MatrixXd(0, 0);
        return;
    }

    // Get data from first GP
    x_ = gp_->getX();
    int n_data = x_.rows();
    int n_gps = gps_.size();
    
    y_ = Eigen::MatrixXd(n_data, n_gps);

    // Collect data from all GPs
    for (size_t i = 0; i < gps_.size(); ++i) {
        const auto& gp_x = gps_[i]->getX();
        const auto& gp_y = gps_[i]->getY();
        
        // Verify that all GPs have the same input data
        if (!x_.isApprox(gp_x)) {
            throw std::runtime_error("All GPs must have the same input data");
        }
        
        y_.col(i) = gp_y;
    }
    
    num_samples_ = n_data;
}

Eigen::VectorXd GaussianProcessOptimization::addContext(
    const Eigen::VectorXd& x, 
    const Eigen::VectorXd& context) const {
    
    if (context.size() == 0) {
        return x;
    }
    
    if (context.size() != num_contexts_) {
        throw std::invalid_argument("Context size must match num_contexts");
    }
    
    Eigen::VectorXd x_with_context(x.size() + context.size());
    x_with_context.head(x.size()) = x;
    x_with_context.tail(context.size()) = context;
    
    return x_with_context;
}

void GaussianProcessOptimization::addDataPoint(
    std::shared_ptr<gp::GaussianProcess> gp,
    const Eigen::VectorXd& x,
    double y,
    const Eigen::VectorXd& context) {
    
    gp->addDataPoint(x, y);
}

void GaussianProcessOptimization::computeAutoScaling() {
    scaling_.clear();
    scaling_.reserve(gps_.size());
    
    for (const auto& gp : gps_) {
        // For auto scaling, we use a simple heuristic
        // In the Python version, this uses kernel diagonal values
        // Here we use a default value that can be tuned
        scaling_.push_back(1.0);  // Simple default
    }
}

} // namespace safeopt