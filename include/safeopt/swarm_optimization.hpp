#pragma once

#include <Eigen/Dense>
#include <vector>
#include <functional>

namespace safeopt {

/**
 * @brief Particle swarm optimization class
 * 
 * General class for constrained swarm optimization.
 * C++ port of the Python SwarmOptimization class.
 */
class SwarmOptimization {
public:
    /**
     * @brief Constructor
     * 
     * @param swarm_size Number of particles in the swarm
     * @param velocity_scale Velocity scaling factors for each dimension
     * @param fitness_func Function that evaluates fitness and safety of positions
     * @param bounds Optional bounds for particle positions
     */
    SwarmOptimization(
        int swarm_size,
        const Eigen::VectorXd& velocity_scale,
        std::function<std::pair<Eigen::VectorXd, Eigen::VectorXi>(const Eigen::MatrixXd&)> fitness_func,
        const std::vector<std::pair<double, double>>& bounds = {}
    );

    /**
     * @brief Initialize swarm with given positions
     * 
     * @param positions Initial positions of particles (rows are particles)
     */
    void initSwarm(const Eigen::MatrixXd& positions);

    /**
     * @brief Run swarm optimization for specified iterations
     * 
     * @param max_iter Maximum number of iterations
     */
    void runSwarm(int max_iter);

    /**
     * @brief Get the best position found so far
     * 
     * @return Best global position
     */
    const Eigen::VectorXd& getGlobalBest() const { return global_best_; }

    /**
     * @brief Get current particle positions
     * 
     * @return Matrix of current positions (rows are particles)
     */
    const Eigen::MatrixXd& getPositions() const { return positions_; }

    /**
     * @brief Get current particle velocities
     * 
     * @return Matrix of current velocities (rows are particles)  
     */
    const Eigen::MatrixXd& getVelocities() const { return velocities_; }

    /**
     * @brief Get best positions found by each particle
     * 
     * @return Matrix of best positions (rows are particles)
     */
    const Eigen::MatrixXd& getBestPositions() const { return best_positions_; }

    /**
     * @brief Get best values found by each particle
     * 
     * @return Vector of best values for each particle
     */
    const Eigen::VectorXd& getBestValues() const { return best_values_; }

    /**
     * @brief Get maximum allowed velocity
     * 
     * @return Maximum velocity vector
     */
    Eigen::VectorXd getMaxVelocity() const { return 10.0 * velocity_scale_; }

protected:
    // Swarm parameters
    int swarm_size_;
    int ndim_;
    double c1_ = 1.0;  // Cognitive parameter
    double c2_ = 1.0;  // Social parameter
    double initial_inertia_ = 1.0;
    double final_inertia_ = 0.1;
    
    // Fitness function
    std::function<std::pair<Eigen::VectorXd, Eigen::VectorXi>(const Eigen::MatrixXd&)> fitness_;
    
    // Bounds
    std::vector<std::pair<double, double>> bounds_;
    
    // Velocity scaling
    Eigen::VectorXd velocity_scale_;
    
    // Swarm state
    Eigen::MatrixXd positions_;       // Current positions
    Eigen::MatrixXd velocities_;      // Current velocities
    Eigen::MatrixXd best_positions_;  // Best positions found by each particle
    Eigen::VectorXd best_values_;     // Best values found by each particle
    Eigen::VectorXd global_best_;     // Best global position

    /**
     * @brief Update particle velocities using PSO rules
     * 
     * @param iteration Current iteration number
     * @param max_iter Maximum number of iterations
     */
    void updateVelocities(int iteration, int max_iter);

    /**
     * @brief Compute inertia weight based on iteration
     * 
     * @param iteration Current iteration
     * @param max_iter Maximum iterations
     * @return Inertia weight
     */
    double computeInertia(int iteration, int max_iter) const;

    /**
     * @brief Clip values to bounds
     * 
     * @param values Matrix to clip
     * @param bounds Bounds to apply
     */
    void clipToBounds(Eigen::MatrixXd& values, 
                     const std::vector<std::pair<double, double>>& bounds) const;
};

} // namespace safeopt