#include "safeopt/safeopt.hpp"
#include "rbf_kernel.h"
#include <iostream>
#include <memory>

using namespace safeopt;

int main() {
    std::cout << "SafeOpt C++ Basic Example" << std::endl;
    
    try {
        // Create a simple 1D optimization problem
        std::vector<std::pair<double, double>> bounds = {{-2.0, 2.0}};
        
        // Create RBF kernel and GP
        auto kernel = std::make_unique<gp::RBFKernel>(1.0, 1.0);
        auto gp = std::make_shared<gp::GaussianProcess>(std::move(kernel), 1e-6);
        
        // Initialize with some safe data
        Eigen::MatrixXd X_init(1, 1);
        X_init << 0.0;
        Eigen::VectorXd Y_init(1);
        Y_init << 1.0;
        gp->fit(X_init, Y_init);
        
        std::vector<std::shared_ptr<gp::GaussianProcess>> gps = {gp};
        std::vector<double> fmin = {0.0};  // Safety threshold
        
        // Create SafeOpt optimizer
        SafeOpt optimizer(gps, fmin, bounds, 2.0);
        
        std::cout << "SafeOpt optimizer created successfully!" << std::endl;
        std::cout << "Current data points: " << optimizer.getT() << std::endl;
        
        // Try to get next point to evaluate
        try {
            Eigen::VectorXd next_point = optimizer.getMaximum();
            std::cout << "Next point to evaluate: " << next_point.transpose() << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Could not get next point: " << e.what() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}