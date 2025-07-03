#include "safeopt/swarm_optimization.hpp"
#include <cassert>
#include <iostream>

using namespace safeopt;

void test_swarm_construction() {
    std::cout << "Testing SwarmOptimization construction..." << std::endl;
    
    int swarm_size = 10;
    Eigen::VectorXd velocity_scale(2);
    velocity_scale << 1.0, 1.0;
    
    auto fitness_func = [](const Eigen::MatrixXd& particles) -> std::pair<Eigen::VectorXd, Eigen::VectorXi> {
        int n = particles.rows();
        Eigen::VectorXd fitness(n);
        Eigen::VectorXi safety(n);
        
        for (int i = 0; i < n; ++i) {
            // Simple fitness: negative of distance from origin
            fitness[i] = -particles.row(i).norm();
            safety[i] = 1;  // All safe
        }
        
        return {fitness, safety};
    };
    
    std::vector<std::pair<double, double>> bounds = {{-2.0, 2.0}, {-2.0, 2.0}};
    
    SwarmOptimization swarm(swarm_size, velocity_scale, fitness_func, bounds);
    
    assert(swarm.getPositions().rows() == swarm_size);
    assert(swarm.getPositions().cols() == 2);
    
    std::cout << "✓ SwarmOptimization construction test passed" << std::endl;
}

void test_swarm_initialization() {
    std::cout << "Testing swarm initialization..." << std::endl;
    
    int swarm_size = 5;
    Eigen::VectorXd velocity_scale(1);
    velocity_scale << 1.0;
    
    auto fitness_func = [](const Eigen::MatrixXd& particles) -> std::pair<Eigen::VectorXd, Eigen::VectorXi> {
        int n = particles.rows();
        Eigen::VectorXd fitness = -particles.col(0).array().square();  // Peak at x=0
        Eigen::VectorXi safety = Eigen::VectorXi::Ones(n);
        return {fitness, safety};
    };
    
    SwarmOptimization swarm(swarm_size, velocity_scale, fitness_func);
    
    // Initialize with specific positions
    Eigen::MatrixXd initial_positions(swarm_size, 1);
    initial_positions << -1.0, -0.5, 0.0, 0.5, 1.0;
    
    swarm.initSwarm(initial_positions);
    
    assert((swarm.getPositions() - initial_positions).norm() < 1e-10);
    assert(swarm.getBestValues().size() == swarm_size);
    
    std::cout << "✓ Swarm initialization test passed" << std::endl;
}

void test_swarm_optimization() {
    std::cout << "Testing swarm optimization..." << std::endl;
    
    int swarm_size = 10;
    Eigen::VectorXd velocity_scale(1);
    velocity_scale << 0.1;
    
    auto fitness_func = [](const Eigen::MatrixXd& particles) -> std::pair<Eigen::VectorXd, Eigen::VectorXi> {
        int n = particles.rows();
        Eigen::VectorXd fitness = -particles.col(0).array().square();  // Peak at x=0
        Eigen::VectorXi safety = Eigen::VectorXi::Ones(n);
        return {fitness, safety};
    };
    
    std::vector<std::pair<double, double>> bounds = {{-2.0, 2.0}};
    SwarmOptimization swarm(swarm_size, velocity_scale, fitness_func, bounds);
    
    // Initialize randomly around x=1 (away from optimum at x=0)
    Eigen::MatrixXd initial_positions = Eigen::MatrixXd::Ones(swarm_size, 1);
    swarm.initSwarm(initial_positions);
    
    double initial_best = swarm.getBestValues().maxCoeff();
    
    // Run optimization
    swarm.runSwarm(20);
    
    double final_best = swarm.getBestValues().maxCoeff();
    
    // Should improve (fitness should increase, getting closer to 0 which is the maximum fitness)
    assert(final_best >= initial_best);
    
    std::cout << "✓ Swarm optimization test passed" << std::endl;
    std::cout << "  Initial best: " << initial_best << std::endl;
    std::cout << "  Final best: " << final_best << std::endl;
    std::cout << "  Best position: " << swarm.getGlobalBest().transpose() << std::endl;
}

int main() {
    std::cout << "Running SwarmOptimization tests..." << std::endl;
    
    try {
        test_swarm_construction();
        test_swarm_initialization();
        test_swarm_optimization();
        
        std::cout << "All SwarmOptimization tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}