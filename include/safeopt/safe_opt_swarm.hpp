#pragma once

#include "safeopt/gaussian_process_optimization.hpp"
#include "safeopt/swarm_optimization.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace safeopt {

/**
 * @brief SafeOpt for larger dimensions using swarm optimization
 * 
 * Scalable version of SafeOpt that uses particle swarm optimization
 * for higher-dimensional problems. Does not support Lipschitz constants
 * or contextual optimization.
 * C++ port of the Python SafeOptSwarm class.
 */
class SafeOptSwarm : public GaussianProcessOptimization {
public:
    /**
     * @brief Constructor
     * 
     * @param gps Vector of Gaussian processes (first is objective, rest are constraints)
     * @param fmin Safety thresholds for each GP  
     * @param bounds Parameter bounds for optimization
     * @param beta Confidence parameter
     * @param threshold Expansion threshold
     * @param scaling Scaling factors for GPs
     * @param swarm_size Number of particles in swarm
     * @param max_iters Maximum iterations for swarm optimization
     */
    SafeOptSwarm(
        std::vector<std::shared_ptr<gp::GaussianProcess>> gps,
        const std::vector<double>& fmin,
        const std::vector<std::pair<double, double>>& bounds,
        double beta = 2.0,
        double threshold = 0.0,
        const std::vector<double>& scaling = {},
        int swarm_size = 20,
        int max_iters = 20
    );

    /**
     * @brief Constructor with beta function
     */
    SafeOptSwarm(
        std::vector<std::shared_ptr<gp::GaussianProcess>> gps,
        const std::vector<double>& fmin,
        const std::vector<std::pair<double, double>>& bounds,
        std::function<double(int)> beta_func,
        double threshold = 0.0,
        const std::vector<double>& scaling = {},
        int swarm_size = 20,
        int max_iters = 20
    );

    /**
     * @brief Get the next point to evaluate
     * 
     * @param context Optional context vector (not supported)
     * @return Next point to evaluate
     */
    Eigen::VectorXd getMaximum(const Eigen::VectorXd& context = Eigen::VectorXd());

    /**
     * @brief Set swarm parameters
     * 
     * @param swarm_size Number of particles
     * @param max_iters Maximum iterations
     */
    void setSwarmParameters(int swarm_size, int max_iters);

protected:
    // Swarm optimization parameters
    int swarm_size_;
    int max_iters_;
    
    // Current best lower bound for maximizers
    double best_lower_bound_;
    
    // Swarm optimizers for different types
    std::unique_ptr<SwarmOptimization> maximizer_swarm_;
    std::unique_ptr<SwarmOptimization> expander_swarm_;

    /**
     * @brief Initialize swarm optimizers
     */
    void initializeSwarms();

    /**
     * @brief Compute particle fitness for maximizer swarm
     * 
     * @param particles Matrix of particle positions
     * @return Pair of (fitness values, safety mask)
     */
    std::pair<Eigen::VectorXd, Eigen::VectorXi> computeMaximizerFitness(
        const Eigen::MatrixXd& particles);

    /**
     * @brief Compute particle fitness for expander swarm
     * 
     * @param particles Matrix of particle positions
     * @return Pair of (fitness values, safety mask)
     */
    std::pair<Eigen::VectorXd, Eigen::VectorXi> computeExpanderFitness(
        const Eigen::MatrixXd& particles);

    /**
     * @brief Compute penalty for constraint violations
     * 
     * @param slack Constraint slack values (positive = satisfied)
     * @return Penalty values
     */
    Eigen::VectorXd computePenalty(const Eigen::MatrixXd& slack) const;

    /**
     * @brief Check if particles are safe according to all constraints
     * 
     * @param particles Matrix of particle positions
     * @return Safety mask for each particle
     */
    Eigen::VectorXi checkSafety(const Eigen::MatrixXd& particles) const;

    /**
     * @brief Get confidence intervals for particles
     * 
     * @param particles Matrix of particle positions
     * @return Matrix with [lower, upper] bounds for each GP and particle
     */
    Eigen::MatrixXd getConfidenceIntervals(const Eigen::MatrixXd& particles) const;

    /**
     * @brief Sample initial particles from safe regions
     * 
     * @param n_particles Number of particles to sample
     * @return Matrix of initial particle positions
     */
    Eigen::MatrixXd sampleInitialParticles(int n_particles) const;

    /**
     * @brief Choose between maximizer and expander optimization
     * 
     * @return Selected next point to evaluate
     */
    Eigen::VectorXd selectNextPoint();

    /**
     * @brief Run maximizer optimization to find points with high objective value
     * 
     * @return Best maximizer point found
     */
    Eigen::VectorXd optimizeMaximizers();

    /**
     * @brief Run expander optimization to expand safe set
     * 
     * @return Best expander point found
     */
    Eigen::VectorXd optimizeExpanders();
};

} // namespace safeopt