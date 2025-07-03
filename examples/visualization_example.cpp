#include "gaussian_process.h"
#include "rbf_kernel.h"
#include "safeopt/safeopt.hpp"
#include "safeopt/visualization.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

// 1D test function with safety constraint
double test_function_1d(double x) {
    return std::sin(x) + 0.1 * x * x - 0.5;
}

// 2D test function: modified sphere function
double test_function_2d(double x, double y) {
    return 1.0 - 0.1 * ((x - 1.0) * (x - 1.0) + (y - 1.0) * (y - 1.0));
}

int main() {
    std::cout << "SafeOpt Visualization Example" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Example 1: 1D Function Optimization with Visualization
    std::cout << "\n1. 1D Function Optimization with Visualization" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    
    {
        // Create Gaussian Process with RBF kernel
        auto kernel = std::make_unique<gp::RBFKernel>(1.0, 1.0);
        auto gp = std::make_shared<gp::GaussianProcess>(std::move(kernel), 0.01);
        
        // Setup 1D bounds
        std::vector<std::pair<double, double>> bounds = {{-3.0, 3.0}};
        
        // Safety threshold
        std::vector<double> fmin = {-0.5};
        
        // Add initial safe point
        Eigen::VectorXd initial_point(1);
        initial_point << 0.0;
        double initial_value = test_function_1d(0.0);
        
        std::cout << "Initial point: " << initial_point.transpose() << std::endl;
        std::cout << "Initial value: " << initial_value << std::endl;
        
        // Fit GP with initial observation
        Eigen::MatrixXd initial_X(1, 1);
        initial_X.row(0) = initial_point.transpose();
        Eigen::VectorXd initial_y(1);
        initial_y << initial_value;
        gp->fit(initial_X, initial_y);
        
        // Create SafeOpt optimizer
        std::vector<std::shared_ptr<gp::GaussianProcess>> gps = {gp};
        safeopt::SafeOpt optimizer(gps, fmin, bounds, 2.0);
        
        // Create discretized 1D grid
        std::vector<double> grid_points;
        for (double x = -3.0; x <= 3.0; x += 0.1) {
            grid_points.push_back(x);
        }
        
        // Convert to Eigen matrix
        Eigen::MatrixXd inputs(grid_points.size(), 1);
        for (size_t i = 0; i < grid_points.size(); ++i) {
            inputs(i, 0) = grid_points[i];
        }
        
        // Set inputs for optimizer
        optimizer.setInputs(inputs);
        
        // Configure visualization
        safeopt::PlotConfig config;
        config.n_samples = 300;
        config.beta = 2.0;
        config.fmin = fmin;
        config.show_safety_threshold = true;
        config.confidence_color = "lightblue";
        config.mean_color = "blue";
        config.latest_point_color = "red";
        
        // Plot initial state
        std::cout << "Plotting initial 1D GP..." << std::endl;
        optimizer.plot();
        
        // Run optimization iterations
        for (int iter = 0; iter < 8; ++iter) {
            std::cout << "\nIteration " << iter + 1 << std::endl;
            
            // Get next point
            Eigen::VectorXd next_point = optimizer.optimize();
            
            // Evaluate function
            double next_value = test_function_1d(next_point(0));
            
            std::cout << "Next point: " << next_point.transpose() << std::endl;
            std::cout << "Function value: " << next_value << std::endl;
            
            // Add observation
            gp->add_data_point(next_point, next_value);
            
            // Plot every few iterations
            if (iter % 2 == 1) {
                std::cout << "Plotting after " << iter + 1 << " iterations..." << std::endl;
                optimizer.plot();
            }
        }
        
        // Plot final state and history
        std::cout << "\nPlotting final 1D optimization state..." << std::endl;
        optimizer.plot();
        
        std::cout << "Plotting 1D optimization history..." << std::endl;
        optimizer.plotHistory();
    }
    
    // Example 2: 2D Function Optimization with Visualization
    std::cout << "\n2. 2D Function Optimization with Visualization" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    
    {
        // Create Gaussian Process with RBF kernel
        auto kernel = std::make_unique<gp::RBFKernel>(1.0, 1.0);
        auto gp = std::make_shared<gp::GaussianProcess>(std::move(kernel), 0.01);
        
        // Setup 2D bounds
        std::vector<std::pair<double, double>> bounds = {{-2.0, 2.0}, {-2.0, 2.0}};
        
        // Safety threshold
        std::vector<double> fmin = {0.0};
        
        // Add initial safe point
        Eigen::VectorXd initial_point(2);
        initial_point << 0.5, 0.5;
        double initial_value = test_function_2d(0.5, 0.5);
        
        std::cout << "Initial point: " << initial_point.transpose() << std::endl;
        std::cout << "Initial value: " << initial_value << std::endl;
        
        // Fit GP with initial observation
        Eigen::MatrixXd initial_X(1, 2);
        initial_X.row(0) = initial_point.transpose();
        Eigen::VectorXd initial_y(1);
        initial_y << initial_value;
        gp->fit(initial_X, initial_y);
        
        // Create SafeOpt optimizer
        std::vector<std::shared_ptr<gp::GaussianProcess>> gps = {gp};
        safeopt::SafeOpt optimizer(gps, fmin, bounds, 2.0);
        
        // Create discretized 2D grid
        std::vector<double> grid_points;
        for (double x = -2.0; x <= 2.0; x += 0.2) {
            for (double y = -2.0; y <= 2.0; y += 0.2) {
                grid_points.push_back(x);
                grid_points.push_back(y);
            }
        }
        
        // Convert to Eigen matrix
        int n_points = grid_points.size() / 2;
        Eigen::MatrixXd inputs(n_points, 2);
        for (int i = 0; i < n_points; ++i) {
            inputs(i, 0) = grid_points[i * 2];
            inputs(i, 1) = grid_points[i * 2 + 1];
        }
        
        // Set inputs for optimizer
        optimizer.setInputs(inputs);
        
        // Configure visualization
        safeopt::PlotConfig config;
        config.n_samples = 400;  // 20x20 grid
        config.beta = 2.0;
        config.fmin = fmin;
        config.show_colorbar = true;
        config.contour_levels = 15;
        config.latest_point_color = "red";
        config.data_point_color = "black";
        
        // Plot initial state (contour)
        std::cout << "Plotting initial 2D GP (contour)..." << std::endl;
        optimizer.plot();
        
        // Run optimization iterations
        for (int iter = 0; iter < 12; ++iter) {
            std::cout << "\nIteration " << iter + 1 << std::endl;
            
            // Get next point
            Eigen::VectorXd next_point = optimizer.optimize();
            
            // Evaluate function
            double next_value = test_function_2d(next_point(0), next_point(1));
            
            std::cout << "Next point: " << next_point.transpose() << std::endl;
            std::cout << "Function value: " << next_value << std::endl;
            
            // Add observation
            gp->add_data_point(next_point, next_value);
            
            // Plot every few iterations
            if (iter % 3 == 2) {
                std::cout << "Plotting after " << iter + 1 << " iterations..." << std::endl;
                optimizer.plot();
                
                // Also plot safe set
                std::cout << "Plotting safe set after " << iter + 1 << " iterations..." << std::endl;
                optimizer.plotSafeSet();
            }
        }
        
        // Plot final state
        std::cout << "\nPlotting final 2D optimization state..." << std::endl;
        optimizer.plot();
        
        // Plot optimization history
        std::cout << "Plotting 2D optimization history..." << std::endl;
        optimizer.plotHistory();
        
        // Plot final safe set
        std::cout << "Plotting final safe set..." << std::endl;
        optimizer.plotSafeSet();
    }
    
    // Example 3: Demonstrate different visualization configurations
    std::cout << "\n3. Visualization Configuration Examples" << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    
    {
        // Create simple 1D example for configuration demo
        auto kernel = std::make_unique<gp::RBFKernel>(1.0, 0.5);
        auto gp = std::make_shared<gp::GaussianProcess>(std::move(kernel), 0.01);
        
        // Add some sample data
        Eigen::MatrixXd X(3, 1);
        Eigen::VectorXd y(3);
        X << -1.0, 0.0, 1.0;
        y << 0.5, 0.8, 0.3;
        gp->fit(X, y);
        
        std::vector<std::pair<double, double>> bounds = {{-2.0, 2.0}};
        std::vector<double> fmin = {0.0};
        
        std::vector<std::shared_ptr<gp::GaussianProcess>> gps = {gp};
        safeopt::SafeOpt optimizer(gps, fmin, bounds, 3.0);
        
        // Create grid
        std::vector<double> grid_points;
        for (double x = -2.0; x <= 2.0; x += 0.05) {
            grid_points.push_back(x);
        }
        Eigen::MatrixXd inputs(grid_points.size(), 1);
        for (size_t i = 0; i < grid_points.size(); ++i) {
            inputs(i, 0) = grid_points[i];
        }
        optimizer.setInputs(inputs);
        
        // Configuration 1: High confidence (narrow intervals)
        safeopt::PlotConfig config1;
        config1.beta = 1.0;
        config1.confidence_color = "green";
        config1.confidence_alpha = 0.2;
        config1.fmin = fmin;
        config1.show_safety_threshold = true;
        
        std::cout << "Plotting basic visualization..." << std::endl;
        optimizer.plot();
    }
    
    std::cout << "\nVisualization examples completed!" << std::endl;
    std::cout << "All plots have been displayed. Check the plot windows." << std::endl;
    
    return 0;
}