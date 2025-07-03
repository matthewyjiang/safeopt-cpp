#include "safeopt/gaussian_process_optimization.hpp"
#include "safeopt/safe_opt.hpp"
#include "safeopt/swarm_optimization.hpp"
#include "safeopt/gp_stub.hpp"
#include <cassert>
#include <iostream>
#include <memory>

using namespace safeopt;

// Test functions from individual test files
void test_basic_construction() {
    std::cout << "Testing basic construction..." << std::endl;
    
    auto gp = std::make_shared<gp::GaussianProcess>();
    std::vector<std::shared_ptr<gp::GaussianProcess>> gps = {gp};
    std::vector<double> fmin = {0.0};
    
    GaussianProcessOptimization opt(gps, fmin);
    
    assert(opt.getT() == 0);  // No initial data
    std::cout << "✓ Basic construction test passed" << std::endl;
}

void test_data_management() {
    std::cout << "Testing data management..." << std::endl;
    
    auto gp = std::make_shared<gp::GaussianProcess>();
    
    // Initialize with some data
    Eigen::MatrixXd X(2, 1);
    X << 0.0, 1.0;
    Eigen::VectorXd Y(2);
    Y << 1.0, 2.0;
    gp->setData(X, Y);
    
    std::vector<std::shared_ptr<gp::GaussianProcess>> gps = {gp};
    std::vector<double> fmin = {0.0};
    
    GaussianProcessOptimization opt(gps, fmin);
    
    assert(opt.getT() == 2);  // Two initial data points
    
    // Add a new data point
    Eigen::VectorXd x_new(1);
    x_new << 2.0;
    Eigen::VectorXd y_new(1);
    y_new << 3.0;
    
    opt.addNewDataPoint(x_new, y_new);
    assert(opt.getT() == 3);  // Now three data points
    
    // Remove last data point
    opt.removeLastDataPoint();
    assert(opt.getT() == 2);  // Back to two data points
    
    std::cout << "✓ Data management test passed" << std::endl;
}

void test_safe_opt_construction() {
    std::cout << "Testing SafeOpt construction..." << std::endl;
    
    auto gp = std::make_shared<gp::GaussianProcess>();
    
    // Initialize with safe data
    Eigen::MatrixXd X(1, 1);
    X << 0.0;
    Eigen::VectorXd Y(1);
    Y << 1.0;
    gp->setData(X, Y);
    
    std::vector<std::shared_ptr<gp::GaussianProcess>> gps = {gp};
    std::vector<double> fmin = {0.0};
    std::vector<std::pair<double, double>> bounds = {{-2.0, 2.0}};
    
    SafeOpt opt(gps, fmin, bounds);
    
    assert(opt.getT() == 1);
    assert(opt.getInputs().rows() > 0);  // Should have discretized inputs
    
    std::cout << "✓ SafeOpt construction test passed" << std::endl;
}

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

int main() {
    std::cout << "Running SafeOpt C++ tests..." << std::endl;
    
    try {
        test_basic_construction();
        test_data_management();
        test_safe_opt_construction();
        test_swarm_construction();
        
        std::cout << std::endl << "All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}