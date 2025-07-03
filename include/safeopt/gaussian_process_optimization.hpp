#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <functional>
#include "safeopt/gp_stub.hpp"

namespace safeopt {

/**
 * @brief Base class for Gaussian Process optimization
 * 
 * Handles common functionality for GP-based optimization algorithms.
 * This is the C++ equivalent of the Python GaussianProcessOptimization class.
 */
class GaussianProcessOptimization {
public:
    /**
     * @brief Constructor
     * 
     * @param gps Vector of Gaussian processes (first is objective, rest are constraints)
     * @param fmin Safety thresholds for each GP
     * @param beta Confidence parameter (constant or function of time)
     * @param num_contexts Number of contextual variables
     * @param threshold Expansion threshold
     * @param scaling Scaling factors for GPs ("auto" or explicit values)
     */
    GaussianProcessOptimization(
        std::vector<std::shared_ptr<gp::GaussianProcess>> gps,
        const std::vector<double>& fmin,
        double beta = 2.0,
        int num_contexts = 0,
        double threshold = 0.0,
        const std::vector<double>& scaling = {}
    );

    /**
     * @brief Constructor with beta function
     */
    GaussianProcessOptimization(
        std::vector<std::shared_ptr<gp::GaussianProcess>> gps,
        const std::vector<double>& fmin,
        std::function<double(int)> beta_func,
        int num_contexts = 0,
        double threshold = 0.0,
        const std::vector<double>& scaling = {}
    );

    virtual ~GaussianProcessOptimization() = default;

    // Properties (getters)
    const Eigen::MatrixXd& getX() const { return x_; }
    const Eigen::MatrixXd& getY() const { return y_; }
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> getData() const { 
        return {x_, y_}; 
    }
    int getT() const { return x_.rows(); }  // Number of observations

    /**
     * @brief Add a new data point to all GPs
     * 
     * @param x Input point
     * @param y Output values (one per GP)
     * @param context Optional context vector
     */
    void addNewDataPoint(const Eigen::VectorXd& x, 
                        const Eigen::VectorXd& y, 
                        const Eigen::VectorXd& context = Eigen::VectorXd());

    /**
     * @brief Remove the last data point from all GPs
     */
    void removeLastDataPoint();

    /**
     * @brief Set parameter bounds for optimization
     * 
     * @param bounds Vector of (min, max) pairs for each parameter
     */
    void setBounds(const std::vector<std::pair<double, double>>& bounds);

    /**
     * @brief Get current bounds
     */
    const std::vector<std::pair<double, double>>& getBounds() const { 
        return bounds_; 
    }

protected:
    // Gaussian processes (first is objective, rest are constraints)
    std::vector<std::shared_ptr<gp::GaussianProcess>> gps_;
    
    // Primary GP (reference to first in gps_)
    std::shared_ptr<gp::GaussianProcess> gp_;
    
    // Safety thresholds
    std::vector<double> fmin_;
    
    // Beta function for confidence intervals
    std::function<double(int)> beta_;
    
    // Scaling factors for GP uncertainties
    std::vector<double> scaling_;
    
    // Algorithm parameters
    double threshold_;
    int num_contexts_;
    int num_samples_;
    
    // Data
    Eigen::MatrixXd x_;  // Input points
    Eigen::MatrixXd y_;  // Output values
    
    // Bounds for optimization
    std::vector<std::pair<double, double>> bounds_;

    /**
     * @brief Initialize data from GPs
     */
    void getInitialXY();

    /**
     * @brief Add context to input vector
     */
    Eigen::VectorXd addContext(const Eigen::VectorXd& x, 
                              const Eigen::VectorXd& context) const;

    /**
     * @brief Add data point to specific GP
     */
    void addDataPoint(std::shared_ptr<gp::GaussianProcess> gp,
                     const Eigen::VectorXd& x,
                     double y,
                     const Eigen::VectorXd& context = Eigen::VectorXd());

    /**
     * @brief Compute automatic scaling from GP kernels
     */
    void computeAutoScaling();
};

} // namespace safeopt