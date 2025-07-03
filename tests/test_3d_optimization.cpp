#include "gaussian_process.h"
#include "rbf_kernel.h"
#include "safeopt/safeopt.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <matplot/matplot.h>
#include <memory>
#include <random>

// 2D test function: modified sphere function with peak at (1, 1)
double test_function_2d(double x, double y) {
  return 1.0 - 0.1 * ((x - 1.0) * (x - 1.0) + (y - 1.0) * (y - 1.0));
}

int main() {
  using namespace matplot;

  std::cout << "Running 2D SafeOpt optimization test (3D visualization)..."
            << std::endl;

  // Create Gaussian Process with RBF kernel
  auto kernel = std::make_unique<gp::RBFKernel>(1.0, 1.0);
  auto gp = std::make_shared<gp::GaussianProcess>(std::move(kernel), 0.001);

  // Setup 2D bounds: [-2, 2] for each dimension
  std::vector<std::pair<double, double>> bounds = {{-2, 2}, {-2, 2}};

  // Safety threshold of 0.0 as requested
  std::vector<double> fmin = {0.0};

  // Add initial safe point near the optimum and fit GP
  Eigen::VectorXd initial_point(2);
  initial_point << -1, 2;
  double initial_value = test_function_2d(0.8, 0.8);

  std::cout << "Initial safe point: (" << initial_point.transpose() << ")"
            << std::endl;
  std::cout << "Initial value: " << initial_value << std::endl;

  // Fit GP with initial observation
  Eigen::MatrixXd initial_X(1, 2);
  initial_X.row(0) = initial_point.transpose();
  Eigen::VectorXd initial_y(1);
  initial_y << initial_value;
  gp->fit(initial_X, initial_y);

  // Create SafeOpt optimizer after fitting GP
  std::vector<std::shared_ptr<gp::GaussianProcess>> gps = {gp};
  safeopt::SafeOpt optimizer(gps, fmin, bounds, 2.0, {}, 0.0);

  // Create discretized 2D grid for optimization
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

  // Set inputs for the optimizer
  optimizer.setInputs(inputs);

  // Run optimization iterations
  std::vector<Eigen::VectorXd> evaluated_points;
  std::vector<double> evaluated_values;

  evaluated_points.push_back(initial_point);
  evaluated_values.push_back(initial_value);

  for (int iter = 0; iter < 10; ++iter) {
    std::cout << "\nIteration " << iter + 1 << std::endl;

    // Get next point to evaluate
    Eigen::VectorXd next_point = optimizer.optimize();

    // Evaluate the function at the next point
    double next_value = test_function_2d(next_point(0), next_point(1));

    std::cout << "Next point: (" << next_point.transpose() << ")" << std::endl;
    std::cout << "Function value: " << next_value << std::endl;

    // Add observation to GP
    gp->add_data_point(next_point, next_value);

    // Store for visualization
    evaluated_points.push_back(next_point);
    evaluated_values.push_back(next_value);

    // Check if we found a point above threshold
    if (next_value > 0.0) {
      std::cout << "Found safe point above threshold!" << std::endl;
    }
  }

  // Find best point
  auto best_iter =
      std::max_element(evaluated_values.begin(), evaluated_values.end());
  int best_idx = std::distance(evaluated_values.begin(), best_iter);

  std::cout << "\nOptimization completed!" << std::endl;
  std::cout << "Best point found: (" << evaluated_points[best_idx].transpose()
            << ")" << std::endl;
  std::cout << "Best value: " << *best_iter << std::endl;
  std::cout << "True optimum is at (1, 1) with value 1.0" << std::endl;

  // Create visualization using mesh plot for the 2D function
  auto [X, Y] = meshgrid(iota(-2, 0.1, 2));
  auto Z = transform(X, Y,
                     [](double x, double y) { return test_function_2d(x, y); });

  // Plot the function surface
  mesh(X, Y, Z);
  hold(on);

  // Plot evaluated points as 3D scatter (x, y, function_value)
  std::vector<double> x_vals, y_vals, z_vals;
  for (size_t i = 0; i < evaluated_points.size(); ++i) {
    x_vals.push_back(evaluated_points[i](0));
    y_vals.push_back(evaluated_points[i](1));
    z_vals.push_back(evaluated_values[i]);
  }

  scatter3(x_vals, y_vals, z_vals)->marker_size(8).marker_color("red");

  // Mark true optimum
  std::vector<double> opt_x = {1.0}, opt_y = {1.0}, opt_z = {1.0};
  scatter3(opt_x, opt_y, opt_z)->marker_size(12).marker_color("green");

  xlabel("X");
  ylabel("Y");
  zlabel("Function Value");
  title("SafeOpt 2D Optimization with 3D Visualization");

  show();

  return 0;
}
