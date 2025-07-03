#pragma once

#include "safeopt/gaussian_process_optimization.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace safeopt {

/**
 * @brief SafeOpt algorithm implementation
 * 
 * Safe Bayesian optimization algorithm that maintains safety constraints
 * while optimizing an objective function.
 * C++ port of the Python SafeOpt class.
 */
class SafeOpt : public GaussianProcessOptimization {
public:
    /**
     * @brief Constructor
     * 
     * @param gps Vector of Gaussian processes (first is objective, rest are constraints)
     * @param fmin Safety thresholds for each GP
     * @param bounds Parameter bounds for optimization
     * @param beta Confidence parameter
     * @param lipschitz Lipschitz constants for each GP (optional)
     * @param threshold Expansion threshold
     * @param scaling Scaling factors for GPs
     */
    SafeOpt(
        std::vector<std::shared_ptr<gp::GaussianProcess>> gps,
        const std::vector<double>& fmin,
        const std::vector<std::pair<double, double>>& bounds,
        double beta = 2.0,
        const std::vector<double>& lipschitz = {},
        double threshold = 0.0,
        const std::vector<double>& scaling = {}
    );

    /**
     * @brief Constructor with beta function
     */
    SafeOpt(
        std::vector<std::shared_ptr<gp::GaussianProcess>> gps,
        const std::vector<double>& fmin,
        const std::vector<std::pair<double, double>>& bounds,
        std::function<double(int)> beta_func,
        const std::vector<double>& lipschitz = {},
        double threshold = 0.0,
        const std::vector<double>& scaling = {}
    );

    /**
     * @brief Get the next point to evaluate
     * 
     * @param context Optional context vector
     * @return Next point to evaluate
     */
    Eigen::VectorXd getMaximum(const Eigen::VectorXd& context = Eigen::VectorXd());

    /**
     * @brief Compute the safe set, expanders, and maximizers
     * 
     * @param full_sets Whether to compute full sets or just next point
     * @param num_expanders Maximum number of expanders to find
     * @param goal Optional goal point for prioritizing expanders
     */
    void computeSets(bool full_sets = false, 
                    int num_expanders = 10,
                    const Eigen::VectorXd& goal = Eigen::VectorXd());

    /**
     * @brief Get the current safe set
     * 
     * @return Boolean mask indicating safe points
     */
    const Eigen::VectorXi& getSafeSet() const { return S_; }

    /**
     * @brief Get the current set of maximizers
     * 
     * @return Boolean mask indicating maximizer points
     */
    const Eigen::VectorXi& getMaximizers() const { return M_; }

    /**
     * @brief Get the current set of expanders
     * 
     * @return Boolean mask indicating expander points
     */
    const Eigen::VectorXi& getExpanders() const { return G_; }

    /**
     * @brief Get discretized input space
     * 
     * @return Matrix of discretized input points
     */
    const Eigen::MatrixXd& getInputs() const { return inputs_; }

    /**
     * @brief Set discretized input space
     * 
     * @param inputs Matrix of input points (rows are points)
     */
    void setInputs(const Eigen::MatrixXd& inputs);

protected:
    // Discretized input space
    Eigen::MatrixXd inputs_;
    
    // Sets (boolean masks over inputs_)
    Eigen::VectorXi S_;  // Safe set
    Eigen::VectorXi M_;  // Maximizers
    Eigen::VectorXi G_;  // Expanders
    
    // Confidence intervals [lower, upper] for each GP at each input
    Eigen::MatrixXd Q_;  // Confidence intervals (columns alternate lower/upper)
    
    // Lipschitz constants
    std::vector<double> lipschitz_;
    bool use_lipschitz_;
    
    // Cached values
    double best_lower_bound_;  // Best lower confidence bound found so far

    /**
     * @brief Initialize the algorithm by computing initial sets
     */
    void initialize();

    /**
     * @brief Compute confidence intervals for all input points
     */
    void computeConfidenceIntervals();

    /**
     * @brief Update the safe set based on confidence intervals
     */
    void updateSafeSet();

    /**
     * @brief Update maximizers within the safe set
     */
    void updateMaximizers();

    /**
     * @brief Update expanders for safe set expansion
     * 
     * @param num_expanders Maximum number of expanders
     * @param goal Optional goal point for prioritizing
     */
    void updateExpanders(int num_expanders, const Eigen::VectorXd& goal);

    /**
     * @brief Check if point can expand safe set using Lipschitz bound
     * 
     * @param point_idx Index of point to check
     * @param gp_idx Index of GP to check
     * @return True if point can expand safe set
     */
    bool canExpandSafeSet(int point_idx, int gp_idx) const;

    /**
     * @brief Get acquisition function value for expander selection
     * 
     * @param point_idx Index of point
     * @return Acquisition value (higher is better)
     */
    double getAcquisitionValue(int point_idx) const;
};

} // namespace safeopt