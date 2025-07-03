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

    // Update internal data matrices first
    int old_size = x_.rows();
    
    // Initialize matrices if this is the first data point
    if (old_size == 0) {
        x_ = Eigen::MatrixXd(1, x_with_context.size());
        y_ = Eigen::MatrixXd(1, gps_.size());
        x_.row(0) = x_with_context;
        y_.row(0) = y.transpose();
    } else {
        x_.conservativeResize(old_size + 1, Eigen::NoChange);
        y_.conservativeResize(old_size + 1, Eigen::NoChange);
        x_.row(old_size) = x_with_context;
        y_.row(old_size) = y.transpose();
    }

    // Add to each GP
    for (size_t i = 0; i < gps_.size(); ++i) {
        if (old_size == 0) {
            // First data point - fit the GP
            gps_[i]->fit(x_, y_.col(i));
        } else {
            // Additional data point - use incremental learning
            addDataPoint(gps_[i], x_with_context, y[i], context);
        }
    }
    
    num_samples_++;
}

void GaussianProcessOptimization::removeLastDataPoint() {
    if (x_.rows() == 0) {
        throw std::runtime_error("No data points to remove");
    }

    // Update internal data matrices
    int new_size = x_.rows() - 1;
    x_.conservativeResize(new_size, Eigen::NoChange);
    y_.conservativeResize(new_size, Eigen::NoChange);
    
    num_samples_--;
    
    // Since the real GP library doesn't support removing data points,
    // we need to refit all GPs with the reduced data
    for (size_t i = 0; i < gps_.size(); ++i) {
        if (new_size > 0) {
            gps_[i]->fit(x_, y_.col(i));
        }
    }
}

void GaussianProcessOptimization::setBounds(
    const std::vector<std::pair<double, double>>& bounds) {
    bounds_ = bounds;
}

void GaussianProcessOptimization::getInitialXY() {
    // Since the real GP library doesn't expose training data,
    // we'll initialize with empty data and let users add data through addNewDataPoint
    x_ = Eigen::MatrixXd(0, 0);
    y_ = Eigen::MatrixXd(0, 0);
    num_samples_ = 0;
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
    
    gp->add_data_point(x, y);
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> GaussianProcessOptimization::getGPData(size_t gp_index) const {
    if (gp_index >= gps_.size()) {
        throw std::out_of_range("GP index out of range");
    }
    
    // Return our locally stored data for the requested GP
    return {x_, y_.col(gp_index)};
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