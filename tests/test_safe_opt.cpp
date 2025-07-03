#include "safeopt/safe_opt.hpp"
#include "safeopt/gp_stub.hpp"
#include <cassert>
#include <iostream>
#include <memory>

using namespace safeopt;

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

void test_sets_computation() {
    std::cout << "Testing sets computation..." << std::endl;
    
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
    
    opt.computeSets();
    
    // Check that sets are computed
    const auto& safe_set = opt.getSafeSet();
    const auto& maximizers = opt.getMaximizers();
    const auto& expanders = opt.getExpanders();
    
    assert(safe_set.size() == opt.getInputs().rows());
    assert(maximizers.size() == opt.getInputs().rows());
    assert(expanders.size() == opt.getInputs().rows());
    
    std::cout << "✓ Sets computation test passed" << std::endl;
}

int main() {
    std::cout << "Running SafeOpt tests..." << std::endl;
    
    try {
        test_safe_opt_construction();
        test_sets_computation();
        
        std::cout << "All SafeOpt tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}