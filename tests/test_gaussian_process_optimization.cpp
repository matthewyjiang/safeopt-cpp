#include "safeopt/gaussian_process_optimization.hpp"
#include "safeopt/gp_stub.hpp"
#include <cassert>
#include <iostream>
#include <memory>

using namespace safeopt;

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

void test_bounds() {
    std::cout << "Testing bounds..." << std::endl;
    
    auto gp = std::make_shared<gp::GaussianProcess>();
    std::vector<std::shared_ptr<gp::GaussianProcess>> gps = {gp};
    std::vector<double> fmin = {0.0};
    
    GaussianProcessOptimization opt(gps, fmin);
    
    std::vector<std::pair<double, double>> bounds = {{-1.0, 1.0}, {-2.0, 2.0}};
    opt.setBounds(bounds);
    
    auto retrieved_bounds = opt.getBounds();
    assert(retrieved_bounds.size() == 2);
    assert(retrieved_bounds[0].first == -1.0);
    assert(retrieved_bounds[0].second == 1.0);
    assert(retrieved_bounds[1].first == -2.0);
    assert(retrieved_bounds[1].second == 2.0);
    
    std::cout << "✓ Bounds test passed" << std::endl;
}

int main() {
    std::cout << "Running GaussianProcessOptimization tests..." << std::endl;
    
    try {
        test_basic_construction();
        test_data_management();
        test_bounds();
        
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}