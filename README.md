# SafeOpt C++

C++ port of the SafeOpt (Safe Bayesian Optimization) library for safe parameter optimization with safety constraints.

## Features

- **SafeOpt**: Exact safe Bayesian optimization algorithm
- **SafeOptSwarm**: Scalable version using particle swarm optimization
- **Visualization**: Real-time plotting with matplot++
- **GP Integration**: Built-in Gaussian process library

## Quick Start

```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Run tests
./test_safeopt
./test_real_function_optimization
```

## Basic Usage

```cpp
#include "safeopt/safeopt.hpp"

// Create GP with RBF kernel
auto kernel = std::make_unique<gp::RBFKernel>(1.0, 1.0);
auto gp = std::make_shared<gp::GaussianProcess>(std::move(kernel), 1e-6);

// Setup SafeOpt
std::vector<std::pair<double, double>> bounds = {{-2.0, 2.0}};
std::vector<std::shared_ptr<gp::GaussianProcess>> gps = {gp};
std::vector<double> fmin = {0.0};  // Safety threshold

safeopt::SafeOpt optimizer(gps, fmin, bounds, 2.0);

// Add initial safe point
Eigen::VectorXd x_safe(1), y_safe(1);
x_safe << 0.0; y_safe << 1.0;
optimizer.addNewDataPoint(x_safe, y_safe);

// Optimization loop
for (int iter = 0; iter < 10; ++iter) {
    Eigen::VectorXd x_next = optimizer.getMaximum();
    double y_next = your_function(x_next);
    
    Eigen::VectorXd y_vec(1);
    y_vec << y_next;
    optimizer.addNewDataPoint(x_next, y_vec);
    
    // Visualize progress
    optimizer.plot();
}
```

## API Reference

### Core Classes

- `safeopt::SafeOpt` - Exact SafeOpt algorithm
- `safeopt::SafeOptSwarm` - Scalable swarm-based version
- `gp::GaussianProcess` - Gaussian process implementation
- `gp::RBFKernel` - RBF kernel for GPs

### Key Methods

- `addNewDataPoint(x, y)` - Add new observation
- `getMaximum()` - Get next point to evaluate
- `getSafeSet()` - Get safe parameter set
- `plot()` - Visualize optimization state

## Dependencies

- Eigen3 (≥3.3)
- CMake 3.10+
- C++17 compiler
- matplot++ (for visualization)

## Testing

```bash
cd build
./test_safeopt                        # Core SafeOpt tests
./test_real_function_optimization     # Real function optimization
./test_visualization                  # Visualization tests
```

## Status

✅ **Completed**
- Core SafeOpt algorithm
- GP integration with RBF kernels
- Visualization with matplot++
- Comprehensive test suite
- Real function optimization examples

## References

Based on the original Python SafeOpt library:
- F. Berkenkamp et al., "Safe Controller Optimization for Quadrotors with Gaussian Processes", ICRA 2016
- F. Berkenkamp et al., "Bayesian Optimization with Safety Constraints", ArXiv 2016

## License

MIT License