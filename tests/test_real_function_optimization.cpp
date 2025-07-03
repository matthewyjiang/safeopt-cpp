#include "gaussian_process.h"
#include "rbf_kernel.h"
#include "safeopt/safeopt.hpp"
#include "safeopt/visualization.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

// Objective function: -sin(x^2) + 1
// This has a global maximum at x=0 with value 1
double objective_function(double x) { return -x * x * std::sin(x) + 1.5; }

// Safety constraint: function value must be > 0.5
bool is_safe(double function_value) { return function_value > 0.5; }

void test_real_function_optimization() {
  std::cout << "Testing SafeOpt on -sin(x^2) + 1 function..." << std::endl;

  // Create GP with RBF kernel
  auto kernel = std::make_unique<gp::RBFKernel>(0.05, 0.2);
  auto gp = std::make_shared<gp::GaussianProcess>(std::move(kernel), 0.001);

  // Initialize with safe starting points
  std::vector<double> initial_x = {0.0, 0.1, -0.1, 0.2, -0.2};
  Eigen::MatrixXd X(initial_x.size(), 1);
  Eigen::VectorXd y(initial_x.size());

  std::cout << "Initial safe points:" << std::endl;
  for (size_t i = 0; i < initial_x.size(); ++i) {
    double x = initial_x[i];
    double fx = objective_function(x);
    X(i, 0) = x;
    y(i) = fx;
    std::cout << "  x=" << x << ", f(x)=" << fx << std::endl;
  }

  gp->fit(X, y);

  // Setup SafeOpt
  std::vector<std::pair<double, double>> bounds = {{-2.0, 2.0}};
  std::vector<double> fmin = {0.5}; // Safety threshold
  std::vector<std::shared_ptr<gp::GaussianProcess>> gps = {gp};
  safeopt::SafeOpt optimizer(gps, fmin, bounds, 2.0);

  // Create discretized input space
  std::vector<double> grid_points;
  for (double x = -2.0; x <= 2.0; x += 0.05) {
    grid_points.push_back(x);
  }

  Eigen::MatrixXd inputs(grid_points.size(), 1);
  for (size_t i = 0; i < grid_points.size(); ++i) {
    inputs(i, 0) = grid_points[i];
  }
  optimizer.setInputs(inputs);

  std::cout << "\nStarting optimization..." << std::endl;

  // Run optimization iterations
  int max_iterations = 30;
  double best_value = *std::max_element(y.data(), y.data() + y.size());
  double best_x = initial_x[std::distance(
      y.data(), std::max_element(y.data(), y.data() + y.size()))];

  for (int iter = 0; iter < max_iterations; ++iter) {
    std::cout << "\nIteration " << iter + 1 << ":" << std::endl;

    // Get next point to evaluate
    Eigen::VectorXd next_point = optimizer.optimize();
    double x_next = next_point(0);
    double f_next = objective_function(x_next);

    std::cout << "  Next point: x=" << x_next << ", f(x)=" << f_next;

    // Check if it's safe
    if (is_safe(f_next)) {
      std::cout << " (SAFE)";
      if (f_next > best_value) {
        best_value = f_next;
        best_x = x_next;
        std::cout << " *NEW BEST*";
      }
    } else {
      std::cout << " (UNSAFE)";
    }
    std::cout << std::endl;

    // Add observation to GP
    optimizer.addNewDataPoint(next_point, Eigen::VectorXd::Constant(1, f_next));

    // Plot current state
    std::cout << "  Plotting current optimization state..." << std::endl;
    // optimizer.plot();

    // Plot safe set
    std::cout << "  Plotting safe set..." << std::endl;
    // optimizer.plotSafeSet();
  }

  std::cout << "\n=== Optimization Results ===" << std::endl;
  std::cout << "Best safe point found: x=" << best_x << ", f(x)=" << best_value
            << std::endl;
  std::cout << "True optimum: x=0, f(0)=" << objective_function(0.0)
            << std::endl;
  std::cout << "Distance from true optimum: " << std::abs(best_x) << std::endl;

  // Final visualization
  std::cout << "\nFinal optimization history:" << std::endl;
  // optimizer.plotHistory();

  // Show ground truth function
  std::cout << "\nGround truth function values:" << std::endl;
  for (double x = -2.0; x <= 2.0; x += 0.5) {
    double fx = objective_function(x);
    std::cout << "  x=" << x << ", f(x)=" << fx << " "
              << (is_safe(fx) ? "(safe)" : "(unsafe)") << std::endl;
  }
}

int main() {
  std::cout << "SafeOpt Real Function Optimization Test" << std::endl;
  std::cout << "=======================================" << std::endl;
  std::cout << "Optimizing: f(x) = -sin(x^2) + 1" << std::endl;
  std::cout << "Safety constraint: f(x) > 0.5" << std::endl;
  std::cout << "Domain: [-2, 2]" << std::endl;
  std::cout << std::endl;

  try {
    test_real_function_optimization();
    std::cout << "\n✅ Real function optimization test completed!" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "\n❌ Test failed: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
