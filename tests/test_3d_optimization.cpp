#include "safeopt/safeopt.hpp"
#include "gaussian_process.h"
#include "rbf_kernel.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>

/**
 * Test function: Modified Branin function with safety constraint
 * Objective: f(x,y,z) = -(a*(y - b*x^2 + c*x - r)^2 + s*(1-t)*cos(x) + s + z^2)
 * Safety constraint: g(x,y,z) = 0.5 - (x^2 + y^2 + z^2) >= 0 (inside unit sphere)
 */
class TestFunction3D {
private:
    // Branin function parameters
    static constexpr double a = 1.0;
    static constexpr double b = 5.1 / (4.0 * M_PI * M_PI);
    static constexpr double c = 5.0 / M_PI;
    static constexpr double r = 6.0;
    static constexpr double s = 10.0;
    static constexpr double t = 1.0 / (8.0 * M_PI);
    
    std::mt19937 rng_;
    std::normal_distribution<double> noise_;

public:
    TestFunction3D() : rng_(42), noise_(0.0, 0.05) {}  // 5% noise
    
    // Objective function (to maximize)
    double objective(double x, double y, double z) {
        double branin = a * std::pow(y - b*x*x + c*x - r, 2) + 
                       s * (1 - t) * std::cos(x) + s;
        return -(branin + 2.0 * z * z) + noise_(rng_);  // Negative for maximization
    }
    
    // Safety constraint (must be >= 0)
    double safety_constraint(double x, double y, double z) {
        return 1.5 - (x*x + y*y + z*z) + 0.1 * noise_(rng_);  // Inside sphere with radius ~1.2
    }
    
    // True optimum location (approximately)
    Eigen::VectorXd true_optimum() {
        Eigen::VectorXd opt(3);
        opt << -M_PI, 12.275, 0.0;  // Branin optimum with z=0
        return opt;
    }
    
    // Evaluate both functions
    Eigen::VectorXd evaluate(const Eigen::VectorXd& x) {
        Eigen::VectorXd result(2);
        result(0) = objective(x(0), x(1), x(2));
        result(1) = safety_constraint(x(0), x(1), x(2));
        return result;
    }
};

void print_progress(int iteration, const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Iter " << std::setw(3) << iteration 
              << ": x=[" << std::setw(6) << x(0) << ", " << std::setw(6) << x(1) 
              << ", " << std::setw(6) << x(2) << "]"
              << " f=" << std::setw(7) << y(0)
              << " g=" << std::setw(7) << y(1)
              << " safe=" << (y(1) >= 0 ? "✓" : "✗") << std::endl;
}

int main() {
    std::cout << "=== SafeOpt 3D Function Optimization Test ===" << std::endl;
    std::cout << "Learning and maximizing a 3D function with safety constraints" << std::endl;
    std::cout << "Objective: Modified Branin function with z^2 penalty" << std::endl;
    std::cout << "Constraint: Must stay within sphere of radius ~1.2" << std::endl << std::endl;

    // Initialize test function
    TestFunction3D func;
    
    // Define bounds: [-2, 2] x [-1, 3] x [-1, 1]
    std::vector<std::pair<double, double>> bounds = {
        {-2.0, 2.0},   // x
        {-1.0, 3.0},   // y  
        {-1.0, 1.0}    // z
    };
    
    // Create Gaussian Processes
    std::cout << "1. Setting up Gaussian Processes..." << std::endl;
    
    // Objective GP
    auto obj_kernel = std::make_unique<gp::RBFKernel>(1.0, 0.5);
    auto obj_gp = std::make_shared<gp::GaussianProcess>(std::move(obj_kernel), 1e-4);
    
    // Constraint GP
    auto cons_kernel = std::make_unique<gp::RBFKernel>(1.0, 0.3);
    auto cons_gp = std::make_shared<gp::GaussianProcess>(std::move(cons_kernel), 1e-4);
    
    std::vector<std::shared_ptr<gp::GaussianProcess>> gps = {obj_gp, cons_gp};
    
    // Safety thresholds
    std::vector<double> fmin = {
        -std::numeric_limits<double>::infinity(),  // No lower bound on objective
        0.0  // Safety constraint must be >= 0
    };
    
    // Find initial safe point
    std::cout << "2. Finding initial safe point..." << std::endl;
    Eigen::VectorXd x_init(3);
    x_init << 0.0, 0.0, 0.0;  // Start at origin (should be safe)
    Eigen::VectorXd y_init = func.evaluate(x_init);
    
    if (y_init(1) < 0) {
        std::cout << "Warning: Initial point is not safe! Constraint value: " << y_init(1) << std::endl;
        // Try a few other points
        for (int i = 0; i < 10; ++i) {
            x_init = 0.1 * Eigen::VectorXd::Random(3);
            y_init = func.evaluate(x_init);
            if (y_init(1) >= 0) break;
        }
    }
    
    std::cout << "Initial point: [" << x_init.transpose() << "]" << std::endl;
    std::cout << "Initial values: f=" << y_init(0) << ", g=" << y_init(1) << std::endl;
    
    // Fit GPs with initial data point
    Eigen::MatrixXd X_init(1, 3);
    X_init.row(0) = x_init.transpose();
    
    Eigen::VectorXd y_obj(1), y_cons(1);
    y_obj(0) = y_init(0);
    y_cons(0) = y_init(1);
    
    obj_gp->fit(X_init, y_obj);
    cons_gp->fit(X_init, y_cons);
    
    // Create SafeOpt optimizer after GPs are fitted
    std::cout << "3. Creating SafeOpt optimizer..." << std::endl;
    safeopt::SafeOpt optimizer(gps, fmin, bounds, 2.0);  // beta = 2.0
    
    // Create custom discretization that includes the initial safe point
    std::vector<int> n_samples = {7, 7, 5};  // More points in x,y dimensions
    Eigen::MatrixXd grid = safeopt::linearly_spaced_combinations(bounds, n_samples);
    
    // Add initial safe point if not already in grid
    bool found_initial = false;
    for (int i = 0; i < grid.rows(); ++i) {
        if ((grid.row(i) - x_init.transpose()).norm() < 1e-6) {
            found_initial = true;
            break;
        }
    }
    
    if (!found_initial) {
        Eigen::MatrixXd new_grid(grid.rows() + 1, grid.cols());
        new_grid.topRows(grid.rows()) = grid;
        new_grid.bottomRows(1) = x_init.transpose();
        grid = new_grid;
    }
    
    optimizer.setInputs(grid);
    
    // Add initial data point to the SafeOpt optimizer
    optimizer.addNewDataPoint(x_init, y_init);
    
    print_progress(0, x_init, y_init);
    
    // Optimization loop
    std::cout << "\n4. Running SafeOpt optimization..." << std::endl;
    std::cout << "Iter   : x=[  x    ,   y   ,   z   ]  f      g      safe" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    
    const int max_iterations = 20;
    double best_objective = y_init(0);
    Eigen::VectorXd best_point = x_init;
    
    for (int iter = 1; iter <= max_iterations; ++iter) {
        try {
            // Alternate between SafeOpt and UCB-only exploration
            bool use_ucb_only = (iter % 3 == 0);  // Every 3rd iteration use UCB-only
            
            // Get next point to evaluate
            Eigen::VectorXd x_next = optimizer.optimize(Eigen::VectorXd(), use_ucb_only);
            
            // Evaluate function
            Eigen::VectorXd y_next = func.evaluate(x_next);
            
            // Add to dataset
            optimizer.addNewDataPoint(x_next, y_next);
            
            // Track best safe point
            if (y_next(1) >= 0 && y_next(0) > best_objective) {
                best_objective = y_next(0);
                best_point = x_next;
            }
            
            // Print progress
            print_progress(iter, x_next, y_next);
            
            // Print diagnostic info every 5 iterations
            if (iter % 5 == 0) {
                auto safe_set = optimizer.getSafeSet();
                auto expanders = optimizer.getExpanders();
                auto maximizers = optimizer.getMaximizers();
                int num_safe = safe_set.sum();
                int num_expanders = expanders.sum();
                int num_maximizers = maximizers.sum();
                
                std::cout << "    → Safe: " << num_safe << ", Maximizers: " << num_maximizers 
                         << ", Expanders: " << num_expanders << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cout << "Error at iteration " << iter << ": " << e.what() << std::endl;
            break;
        }
    }
    
    // Results
    std::cout << "\n=== Optimization Results ===" << std::endl;
    std::cout << "Best safe point found: [" << best_point.transpose() << "]" << std::endl;
    std::cout << "Best objective value: " << best_objective << std::endl;
    
    // Compare with true optimum (if in safe region)
    Eigen::VectorXd true_opt = func.true_optimum();
    double true_safety = func.safety_constraint(true_opt(0), true_opt(1), true_opt(2));
    std::cout << "\nTrue optimum: [" << true_opt.transpose() << "]" << std::endl;
    std::cout << "True optimum safety: " << true_safety << " (safe: " << (true_safety >= 0 ? "✓" : "✗") << ")" << std::endl;
    
    if (true_safety >= 0) {
        double true_obj = func.objective(true_opt(0), true_opt(1), true_opt(2));
        std::cout << "True optimum objective: " << true_obj << std::endl;
        std::cout << "Optimization gap: " << (true_obj - best_objective) << std::endl;
    }
    
    // Safe set statistics
    auto safe_set = optimizer.getSafeSet();
    auto maximizers = optimizer.getMaximizers();
    auto expanders = optimizer.getExpanders();
    
    std::cout << "\nFinal statistics:" << std::endl;
    std::cout << "Safe set size: " << safe_set.sum() << " / " << safe_set.size() << std::endl;
    std::cout << "Maximizers: " << maximizers.sum() << std::endl;
    std::cout << "Expanders: " << expanders.sum() << std::endl;
    
    std::cout << "\n✓ 3D optimization test completed!" << std::endl;
    
    return 0;
}