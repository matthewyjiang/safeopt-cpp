#pragma once

#include <Eigen/Dense>
#include <vector>
#include <functional>

namespace safeopt {

/**
 * @brief Generate linearly spaced combinations of points within bounds
 * 
 * @param bounds Vector of pairs (min, max) for each dimension
 * @param num_samples Number of samples per dimension
 * @return Matrix where each row is a sample point
 */
Eigen::MatrixXd linearly_spaced_combinations(
    const std::vector<std::pair<double, double>>& bounds,
    const std::vector<int>& num_samples);

/**
 * @brief Generate linearly spaced combinations with same number of samples per dimension
 * 
 * @param bounds Vector of pairs (min, max) for each dimension  
 * @param num_samples Number of samples per dimension (same for all dimensions)
 * @return Matrix where each row is a sample point
 */
Eigen::MatrixXd linearly_spaced_combinations(
    const std::vector<std::pair<double, double>>& bounds,
    int num_samples);

/**
 * @brief Sample a GP function for testing purposes
 * 
 * This creates a sample function from a GP prior that can be used
 * for testing optimization algorithms.
 * 
 * @param bounds Bounds for input domain
 * @param noise_var Noise variance for observations
 * @param num_samples Number of discretization points
 * @return Function that evaluates the sampled GP function
 */
std::function<Eigen::VectorXd(const Eigen::MatrixXd&, bool)> 
sample_gp_function(
    const std::vector<std::pair<double, double>>& bounds,
    double noise_var,
    int num_samples);

} // namespace safeopt