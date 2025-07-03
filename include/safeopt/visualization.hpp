#pragma once

#include <Eigen/Dense>
#include <matplot/matplot.h>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <utility>

#include "gaussian_process.h"
#include "safe_opt.hpp"

namespace safeopt {

/**
 * @brief Configuration structure for SafeOpt visualization
 */
struct PlotConfig {
    // Sampling parameters
    int n_samples = 1000;           // Number of sample points for plotting
    double beta = 3.0;              // Confidence interval scaling factor
    bool plot_3d = false;           // Use 3D surface plot for 2D functions
    bool show_colorbar = true;      // Show colorbar in contour plots
    bool show_safety_threshold = true; // Show safety threshold lines
    
    // Safety parameters
    std::vector<double> fmin;       // Safety threshold values
    std::vector<std::pair<int, double>> fixed_inputs; // Fixed input dimensions
    
    // Visual styling
    std::string confidence_color = "blue";
    double confidence_alpha = 0.3;
    std::string mean_color = "blue";
    std::string data_point_color = "black";
    std::string latest_point_color = "red";
    int marker_size = 10;
    
    // Plot dimensions
    double line_width = 2.0;
    std::string colormap = "viridis";
    int contour_levels = 20;
};

/**
 * @brief Visualization class for SafeOpt optimization results
 */
class SafeOptVisualizer {
public:
    /**
     * @brief Constructor
     */
    SafeOptVisualizer() = default;
    
    /**
     * @brief Plot 1D Gaussian Process with confidence intervals
     * @param gp The Gaussian Process to plot
     * @param inputs Input points for evaluation
     * @param config Plotting configuration
     */
    void plot_1d_gp(const gp::GaussianProcess& gp, 
                    const std::vector<double>& inputs,
                    const PlotConfig& config = PlotConfig{});
    
    /**
     * @brief Plot 2D Gaussian Process as contour plot
     * @param gp The Gaussian Process to plot
     * @param bounds Input bounds for each dimension
     * @param config Plotting configuration
     */
    void plot_contour_gp(const gp::GaussianProcess& gp,
                        const std::vector<std::pair<double, double>>& bounds,
                        const PlotConfig& config = PlotConfig{});
    
    /**
     * @brief Plot 2D Gaussian Process as 3D surface
     * @param gp The Gaussian Process to plot
     * @param bounds Input bounds for each dimension
     * @param config Plotting configuration
     */
    void plot_3d_gp(const gp::GaussianProcess& gp,
                   const std::vector<std::pair<double, double>>& bounds,
                   const PlotConfig& config = PlotConfig{});
    
    /**
     * @brief Main plotting interface - auto-detects dimensionality
     * @param optimizer The SafeOpt optimizer
     * @param config Plotting configuration
     */
    void plot(const SafeOpt& optimizer,
              const PlotConfig& config = PlotConfig{});
    
    /**
     * @brief Plot optimization history
     * @param optimizer The SafeOpt optimizer
     * @param config Plotting configuration
     */
    void plot_history(const SafeOpt& optimizer,
                     const PlotConfig& config = PlotConfig{});
    
    /**
     * @brief Plot safe set and expanders
     * @param optimizer The SafeOpt optimizer
     * @param config Plotting configuration
     */
    void plot_safe_set(const SafeOpt& optimizer,
                      const PlotConfig& config = PlotConfig{});

private:
    /**
     * @brief Generate 1D input grid for plotting
     * @param bounds Input bounds
     * @param n_samples Number of samples
     * @return Vector of input points
     */
    std::vector<double> generate_1d_grid(const std::pair<double, double>& bounds,
                                         int n_samples) const;
    
    /**
     * @brief Generate 2D input grid for plotting
     * @param bounds Input bounds for each dimension
     * @param n_samples Number of samples per dimension
     * @return Pair of X and Y coordinate matrices
     */
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
    generate_2d_grid(const std::vector<std::pair<double, double>>& bounds,
                     int n_samples) const;
    
    /**
     * @brief Convert 2D grid to input points for GP prediction
     * @param x_grid X coordinate grid
     * @param y_grid Y coordinate grid
     * @return Matrix of input points
     */
    Eigen::MatrixXd grid_to_points(const std::vector<std::vector<double>>& x_grid,
                                   const std::vector<std::vector<double>>& y_grid) const;
    
    /**
     * @brief Plot data points on current axes
     * @param X Input data points
     * @param y Output values
     * @param config Plotting configuration
     */
    void plot_data_points(const Eigen::MatrixXd& X,
                         const Eigen::VectorXd& y,
                         const PlotConfig& config) const;
    
    /**
     * @brief Plot safety threshold line
     * @param x_range X-axis range
     * @param fmin Safety threshold value
     */
    void plot_safety_threshold(const std::pair<double, double>& x_range,
                              double fmin) const;
    
    /**
     * @brief Calculate confidence intervals
     * @param mean Mean predictions
     * @param variance Prediction variances
     * @param beta Confidence scaling factor
     * @return Pair of lower and upper bounds
     */
    std::pair<std::vector<double>, std::vector<double>> 
    calculate_confidence_intervals(const std::vector<double>& mean,
                                  const std::vector<double>& variance,
                                  double beta) const;
};

/**
 * @brief Utility functions for visualization
 */
namespace viz_utils {

/**
 * @brief Generate linearly spaced combinations of inputs
 * @param bounds Input bounds for each dimension
 * @param num_samples Number of samples per dimension
 * @return Matrix of input combinations
 */
Eigen::MatrixXd linearly_spaced_combinations(
    const std::vector<std::pair<double, double>>& bounds,
    int num_samples);

/**
 * @brief Generate linearly spaced combinations with different samples per dimension
 * @param bounds Input bounds for each dimension
 * @param num_samples Number of samples for each dimension
 * @return Matrix of input combinations
 */
Eigen::MatrixXd linearly_spaced_combinations(
    const std::vector<std::pair<double, double>>& bounds,
    const std::vector<int>& num_samples);

/**
 * @brief Create a colormap for visualization
 * @param values Values to map to colors
 * @param colormap_name Name of the colormap
 * @return Vector of color values
 */
std::vector<std::vector<double>> create_colormap(const std::vector<double>& values,
                                                const std::string& colormap_name = "viridis");

} // namespace viz_utils

} // namespace safeopt