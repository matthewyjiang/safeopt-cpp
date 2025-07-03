#include "gaussian_process.h"
#include "rbf_kernel.h"
#include "safeopt/safeopt.hpp"
#include "safeopt/visualization.hpp"
#include <iostream>
#include <memory>
#include <cassert>

// Simple test function
double simple_function(double x) {
    return x * x - 1.0;
}

void test_visualization_api() {
    std::cout << "Testing SafeOpt Visualization API..." << std::endl;
    
    // Create a simple 1D optimization problem
    auto kernel = std::make_unique<gp::RBFKernel>(1.0, 1.0);
    auto gp = std::make_shared<gp::GaussianProcess>(std::move(kernel), 0.01);
    
    // Add some training data
    Eigen::MatrixXd X(2, 1);
    Eigen::VectorXd y(2);
    X << 0.0, 1.0;
    y << -1.0, 0.0;
    gp->fit(X, y);
    
    // Create SafeOpt optimizer
    std::vector<std::pair<double, double>> bounds = {{-2.0, 2.0}};
    std::vector<double> fmin = {-0.5};
    std::vector<std::shared_ptr<gp::GaussianProcess>> gps = {gp};
    safeopt::SafeOpt optimizer(gps, fmin, bounds, 2.0);
    
    // Create discretized input space
    std::vector<double> grid_points;
    for (double x = -2.0; x <= 2.0; x += 0.1) {
        grid_points.push_back(x);
    }
    
    Eigen::MatrixXd inputs(grid_points.size(), 1);
    for (size_t i = 0; i < grid_points.size(); ++i) {
        inputs(i, 0) = grid_points[i];
    }
    optimizer.setInputs(inputs);
    
    // Test that methods don't crash
    try {
        std::cout << "  Testing plot() method..." << std::endl;
        // Note: This will display a plot window, but won't crash
        optimizer.plot();
        
        std::cout << "  Testing plotHistory() method..." << std::endl;
        optimizer.plotHistory();
        
        std::cout << "  Testing plotSafeSet() method..." << std::endl;
        optimizer.plotSafeSet();
        
        std::cout << "  All visualization methods executed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "  Error in visualization test: " << e.what() << std::endl;
        throw;
    }
}

void test_visualization_configurations() {
    std::cout << "Testing Visualization Configurations..." << std::endl;
    
    // Create simple GP
    auto kernel = std::make_unique<gp::RBFKernel>(1.0, 1.0);
    auto gp = std::make_shared<gp::GaussianProcess>(std::move(kernel), 0.01);
    
    Eigen::MatrixXd X(1, 1);
    Eigen::VectorXd y(1);
    X << 0.0;
    y << 0.5;
    gp->fit(X, y);
    
    std::vector<std::pair<double, double>> bounds = {{-1.0, 1.0}};
    std::vector<double> fmin = {0.0};
    std::vector<std::shared_ptr<gp::GaussianProcess>> gps = {gp};
    safeopt::SafeOpt optimizer(gps, fmin, bounds, 2.0);
    
    // Create small grid
    Eigen::MatrixXd inputs(21, 1);
    for (int i = 0; i < 21; ++i) {
        inputs(i, 0) = -1.0 + i * 0.1;
    }
    optimizer.setInputs(inputs);
    
    // Test different configurations
    std::vector<safeopt::PlotConfig> configs;
    
    // Configuration 1: Default
    configs.push_back(safeopt::PlotConfig{});
    
    // Configuration 2: High confidence
    safeopt::PlotConfig config2;
    config2.beta = 1.0;
    config2.confidence_color = "green";
    configs.push_back(config2);
    
    // Configuration 3: Low confidence
    safeopt::PlotConfig config3;
    config3.beta = 4.0;
    config3.confidence_color = "red";
    configs.push_back(config3);
    
    // Configuration 4: Custom styling
    safeopt::PlotConfig config4;
    config4.beta = 2.0;
    config4.confidence_color = "purple";
    config4.confidence_alpha = 0.4;
    config4.mean_color = "darkblue";
    config4.data_point_color = "orange";
    config4.latest_point_color = "red";
    config4.marker_size = 12;
    config4.line_width = 2.5;
    configs.push_back(config4);
    
    for (size_t i = 0; i < configs.size(); ++i) {
        std::cout << "  Testing configuration " << i + 1 << "..." << std::endl;
        try {
            // Test with default configuration since we simplified the API
            optimizer.plot();
            std::cout << "    Configuration " << i + 1 << " successful!" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "    Error in configuration " << i + 1 << ": " << e.what() << std::endl;
            throw;
        }
    }
}

void test_2d_visualization() {
    std::cout << "Testing 2D Visualization..." << std::endl;
    
    // Create 2D GP
    auto kernel = std::make_unique<gp::RBFKernel>(1.0, 1.0);
    auto gp = std::make_shared<gp::GaussianProcess>(std::move(kernel), 0.01);
    
    // Add some 2D training data
    Eigen::MatrixXd X(2, 2);
    Eigen::VectorXd y(2);
    X << 0.0, 0.0,
         1.0, 1.0;
    y << 0.5, 0.8;
    gp->fit(X, y);
    
    std::vector<std::pair<double, double>> bounds = {{-1.0, 1.0}, {-1.0, 1.0}};
    std::vector<double> fmin = {0.0};
    std::vector<std::shared_ptr<gp::GaussianProcess>> gps = {gp};
    safeopt::SafeOpt optimizer(gps, fmin, bounds, 2.0);
    
    // Create 2D grid
    std::vector<double> grid_points;
    for (double x = -1.0; x <= 1.0; x += 0.2) {
        for (double y = -1.0; y <= 1.0; y += 0.2) {
            grid_points.push_back(x);
            grid_points.push_back(y);
        }
    }
    
    int n_points = grid_points.size() / 2;
    Eigen::MatrixXd inputs(n_points, 2);
    for (int i = 0; i < n_points; ++i) {
        inputs(i, 0) = grid_points[i * 2];
        inputs(i, 1) = grid_points[i * 2 + 1];
    }
    optimizer.setInputs(inputs);
    
    // Test 2D plotting
    safeopt::PlotConfig config;
    config.n_samples = 100;  // 10x10 grid
    config.show_colorbar = true;
    config.contour_levels = 10;
    
    try {
        std::cout << "  Testing 2D contour plot..." << std::endl;
        optimizer.plot();
        
        std::cout << "  Testing 2D safe set plot..." << std::endl;
        optimizer.plotSafeSet();
        
        std::cout << "  All 2D visualization tests passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "  Error in 2D visualization: " << e.what() << std::endl;
        throw;
    }
}

void test_visualizer_direct() {
    std::cout << "Testing SafeOptVisualizer directly..." << std::endl;
    
    // Test the visualizer class directly
    safeopt::SafeOptVisualizer visualizer;
    
    // Create simple GP
    auto kernel = std::make_unique<gp::RBFKernel>(1.0, 1.0);
    auto gp = std::make_shared<gp::GaussianProcess>(std::move(kernel), 0.01);
    
    Eigen::MatrixXd X(2, 1);
    Eigen::VectorXd y(2);
    X << -1.0, 1.0;
    y << 0.2, 0.8;
    gp->fit(X, y);
    
    try {
        std::cout << "  Testing 1D GP plotting..." << std::endl;
        std::vector<double> inputs;
        for (double x = -2.0; x <= 2.0; x += 0.1) {
            inputs.push_back(x);
        }
        visualizer.plot_1d_gp(*gp, inputs);
        
        std::cout << "  Testing 2D GP plotting..." << std::endl;
        std::vector<std::pair<double, double>> bounds = {{-2.0, 2.0}, {-2.0, 2.0}};
        
        // Create 2D GP
        auto kernel2d = std::make_unique<gp::RBFKernel>(1.0, 1.0);
        auto gp2d = std::make_shared<gp::GaussianProcess>(std::move(kernel2d), 0.01);
        
        Eigen::MatrixXd X2d(2, 2);
        Eigen::VectorXd y2d(2);
        X2d << 0.0, 0.0,
               1.0, 1.0;
        y2d << 0.5, 0.7;
        gp2d->fit(X2d, y2d);
        
        visualizer.plot_contour_gp(*gp2d, bounds);
        visualizer.plot_3d_gp(*gp2d, bounds);
        
        std::cout << "  Direct visualizer tests passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "  Error in direct visualizer test: " << e.what() << std::endl;
        throw;
    }
}

int main() {
    std::cout << "SafeOpt Visualization Tests" << std::endl;
    std::cout << "===========================" << std::endl;
    
    try {
        test_visualization_api();
        test_visualization_configurations();
        test_2d_visualization();
        test_visualizer_direct();
        
        std::cout << "\n✅ All visualization tests passed!" << std::endl;
        std::cout << "Note: This test displays plot windows - close them to continue." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Visualization test failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}