# SafeOpt C++

C++ port of the SafeOpt (Safe Bayesian Optimization) library, originally written in Python.

## Overview

SafeOpt is a library for Safe Bayesian Optimization - automatically optimizing performance measures subject to safety constraints by adapting parameters. This C++ implementation provides the core algorithms with modern C++ design patterns and high performance.

## Features

- **GaussianProcessOptimization**: Base class for GP-based optimization
- **SafeOpt**: Exact SafeOpt algorithm for safe Bayesian optimization
- **SafeOptSwarm**: Scalable version using particle swarm optimization for higher dimensions
- **SwarmOptimization**: General particle swarm optimization framework
- **Utilities**: Helper functions for sampling and data generation

## Dependencies

- **Eigen3**: Linear algebra library
- **CMake 3.10+**: Build system
- **C++17 compiler**: Modern C++ features

### Future Dependencies

- **gaussian-process library**: Will integrate with https://github.com/matthewyjiang/gaussian-process for production use

## Installation

### Ubuntu/Debian

```bash
sudo apt-get install libeigen3-dev cmake build-essential
```

### Build from Source

```bash
git clone <repository-url>
cd safeopt-cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

## Usage

### Basic Example

```cpp
#include "safeopt/safeopt.hpp"

int main() {
    // Define optimization bounds
    std::vector<std::pair<double, double>> bounds = {{-2.0, 2.0}};
    
    // Create Gaussian process (placeholder implementation)
    auto gp = std::make_shared<gp::GaussianProcess>();
    
    // Initialize with safe data point
    Eigen::MatrixXd X_init(1, 1);
    X_init << 0.0;
    Eigen::VectorXd Y_init(1);
    Y_init << 1.0;
    gp->setData(X_init, Y_init);
    
    // Setup SafeOpt
    std::vector<std::shared_ptr<gp::GaussianProcess>> gps = {gp};
    std::vector<double> fmin = {0.0};  // Safety threshold
    
    safeopt::SafeOpt optimizer(gps, fmin, bounds, 2.0);
    
    // Get next point to evaluate
    Eigen::VectorXd next_point = optimizer.getMaximum();
    
    return 0;
}
```

### SafeOptSwarm for High Dimensions

```cpp
#include "safeopt/safe_opt_swarm.hpp"

// For problems with many dimensions, use SafeOptSwarm
safeopt::SafeOptSwarm optimizer(gps, fmin, bounds, 
                               2.0,    // beta
                               0.0,    // threshold
                               {},     // auto scaling
                               20,     // swarm size
                               20);    // max iterations

Eigen::VectorXd next_point = optimizer.getMaximum();
```

## Testing

Run the test suite:

```bash
cd build
./test_safeopt
```

Run the example:

```bash
cd build
./safeopt_example
```

## Current Status

This is a **functional prototype** of the C++ port with the following status:

### ✅ Completed
- [x] Project structure and build system
- [x] Core class interfaces (SafeOpt, SafeOptSwarm, GaussianProcessOptimization)
- [x] Particle swarm optimization implementation
- [x] Basic GP stub for compilation and testing
- [x] Utility functions for sampling and combinations
- [x] Basic test suite and examples
- [x] Documentation and build instructions

### 🚧 In Progress / TODO
- [ ] Integration with real Gaussian Process library
- [ ] Complete SafeOpt algorithm implementation (Lipschitz bounds, full acquisition functions)
- [ ] Advanced acquisition functions (UCB, EI, etc.)
- [ ] Contextual optimization support
- [ ] Performance optimizations
- [ ] Comprehensive test coverage
- [ ] Python binding support
- [ ] Visualization capabilities

## Architecture

```
include/safeopt/
├── safeopt.hpp                          # Main header
├── gaussian_process_optimization.hpp    # Base GP optimization class
├── safe_opt.hpp                        # Exact SafeOpt algorithm
├── safe_opt_swarm.hpp                  # Scalable swarm-based SafeOpt
├── swarm_optimization.hpp              # General swarm optimization
├── utilities.hpp                       # Utility functions
└── gp_stub.hpp                        # GP interface (temporary)

src/
├── gaussian_process_optimization.cpp
├── safe_opt.cpp
├── safe_opt_swarm.cpp
├── swarm_optimization.cpp
└── utilities.cpp
```

## Contributing

This is an early-stage port. Contributions are welcome, especially:

1. Integration with the gaussian-process library
2. Algorithm completeness and correctness
3. Performance optimizations
4. Test coverage improvements
5. Documentation enhancements

## References

Based on the original Python SafeOpt library:

- F. Berkenkamp, A. P. Schoellig, A. Krause, "Safe Controller Optimization for Quadrotors with Gaussian Processes", ICRA 2016
- F. Berkenkamp, A. Krause, A. P. Schoellig, "Bayesian Optimization with Safety Constraints: Safe and Automatic Parameter Tuning in Robotics", ArXiv 2016

## License

MIT License (same as original Python version)