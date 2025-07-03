#include "safeopt/visualization.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace safeopt {

void SafeOptVisualizer::plot_1d_gp(const gp::GaussianProcess& gp, 
                                  const std::vector<double>& inputs,
                                  const PlotConfig& config) {
    using namespace matplot;
    
    // Convert inputs to Eigen format for prediction
    Eigen::MatrixXd X_test(inputs.size(), 1);
    for (size_t i = 0; i < inputs.size(); ++i) {
        X_test(i, 0) = inputs[i];
    }
    
    // Get predictions with uncertainty
    auto [mean_vec, var_vec] = gp.predict(X_test, true);
    
    // Convert to std::vector
    std::vector<double> mean(mean_vec.data(), mean_vec.data() + mean_vec.size());
    std::vector<double> variance(var_vec.data(), var_vec.data() + var_vec.size());
    
    // Calculate confidence intervals
    auto [lower_bound, upper_bound] = calculate_confidence_intervals(mean, variance, config.beta);
    
    // Create new figure
    figure();
    
    // Plot confidence interval as filled area
    std::vector<double> x_fill, y_fill;
    for (size_t i = 0; i < inputs.size(); ++i) {
        x_fill.push_back(inputs[i]);
        y_fill.push_back(upper_bound[i]);
    }
    for (int i = inputs.size() - 1; i >= 0; --i) {
        x_fill.push_back(inputs[i]);
        y_fill.push_back(lower_bound[i]);
    }
    
    auto conf_plot = fill(x_fill, y_fill);
    conf_plot->color(config.confidence_color);
    conf_plot->display_name("Confidence Interval");
    
    hold(on);
    
    // Plot mean function
    auto mean_plot = matplot::plot(inputs, mean);
    mean_plot->color(config.mean_color);
    mean_plot->line_width(config.line_width);
    mean_plot->display_name("GP Mean");
    
    // Plot data points if GP has training data
    if (gp.is_fitted()) {
        auto X_train = gp.getX();
        auto y_train = gp.getY();
        
        if (X_train.cols() == 1) {  // Only plot if 1D
            std::vector<double> x_data, y_data;
            for (int i = 0; i < X_train.rows(); ++i) {
                x_data.push_back(X_train(i, 0));
                y_data.push_back(y_train(i));
            }
            
            auto data_plot = scatter(x_data, y_data);
            data_plot->marker_size(config.marker_size);
            data_plot->color(config.data_point_color);
            data_plot->color(config.data_point_color);
            data_plot->display_name("Training Data");
            
            // Highlight latest point
            if (!x_data.empty()) {
                std::vector<double> latest_x = {x_data.back()};
                std::vector<double> latest_y = {y_data.back()};
                auto latest_plot = scatter(latest_x, latest_y);
                latest_plot->marker_size(config.marker_size + 2);
                latest_plot->color(config.latest_point_color);
                latest_plot->color(config.latest_point_color);
                latest_plot->display_name("Latest Point");
            }
        }
    }
    
    // Plot safety threshold if specified
    if (config.show_safety_threshold && !config.fmin.empty()) {
        std::pair<double, double> x_range = {*std::min_element(inputs.begin(), inputs.end()),
                                           *std::max_element(inputs.begin(), inputs.end())};
        plot_safety_threshold(x_range, config.fmin[0]);
    }
    
    xlabel("Input");
    ylabel("Output");
    title("SafeOpt 1D Gaussian Process");
    legend();
    grid(on);
    
    show();
}

void SafeOptVisualizer::plot_contour_gp(const gp::GaussianProcess& gp,
                                       const std::vector<std::pair<double, double>>& bounds,
                                       const PlotConfig& config) {
    using namespace matplot;
    
    // Generate 2D grid
    auto [x_grid, y_grid] = generate_2d_grid(bounds, config.n_samples);
    
    // Convert grid to prediction points
    Eigen::MatrixXd X_test = grid_to_points(x_grid, y_grid);
    
    // Get predictions
    auto [mean_vec, var_vec] = gp.predict(X_test, true);
    
    // Reshape predictions to grid
    std::vector<std::vector<double>> Z(x_grid.size(), std::vector<double>(x_grid[0].size()));
    int idx = 0;
    for (size_t i = 0; i < x_grid.size(); ++i) {
        for (size_t j = 0; j < x_grid[i].size(); ++j) {
            Z[i][j] = mean_vec(idx++);
        }
    }
    
    // Create contour plot
    figure();
    auto contour_plot = contour(x_grid, y_grid, Z, config.contour_levels);
    contour_plot->line_width(1.0);
    
    if (config.show_colorbar) {
        colorbar();
    }
    
    hold(on);
    
    // Plot data points if GP has training data
    if (gp.is_fitted()) {
        auto X_train = gp.getX();
        auto y_train = gp.getY();
        
        if (X_train.cols() == 2) {  // Only plot if 2D
            std::vector<double> x_data, y_data;
            for (int i = 0; i < X_train.rows(); ++i) {
                x_data.push_back(X_train(i, 0));
                y_data.push_back(X_train(i, 1));
            }
            
            auto data_plot = scatter(x_data, y_data);
            data_plot->marker_size(config.marker_size);
            data_plot->color(config.data_point_color);
            data_plot->color(config.data_point_color);
            data_plot->display_name("Training Data");
            
            // Highlight latest point
            if (!x_data.empty()) {
                std::vector<double> latest_x = {x_data.back()};
                std::vector<double> latest_y = {y_data.back()};
                auto latest_plot = scatter(latest_x, latest_y);
                latest_plot->marker_size(config.marker_size + 2);
                latest_plot->color(config.latest_point_color);
                latest_plot->color(config.latest_point_color);
                latest_plot->display_name("Latest Point");
            }
        }
    }
    
    xlabel("Input 1");
    ylabel("Input 2");
    title("SafeOpt 2D Gaussian Process (Contour)");
    legend();
    
    show();
}

void SafeOptVisualizer::plot_3d_gp(const gp::GaussianProcess& gp,
                                  const std::vector<std::pair<double, double>>& bounds,
                                  const PlotConfig& config) {
    using namespace matplot;
    
    // Generate 2D grid
    auto [x_grid, y_grid] = generate_2d_grid(bounds, config.n_samples);
    
    // Convert grid to prediction points
    Eigen::MatrixXd X_test = grid_to_points(x_grid, y_grid);
    
    // Get predictions
    auto [mean_vec, var_vec] = gp.predict(X_test, true);
    
    // Reshape predictions to grid
    std::vector<std::vector<double>> Z(x_grid.size(), std::vector<double>(x_grid[0].size()));
    int idx = 0;
    for (size_t i = 0; i < x_grid.size(); ++i) {
        for (size_t j = 0; j < x_grid[i].size(); ++j) {
            Z[i][j] = mean_vec(idx++);
        }
    }
    
    // Create 3D surface plot
    figure();
    auto surface_plot = surf(x_grid, y_grid, Z);
    
    if (config.show_colorbar) {
        colorbar();
    }
    
    hold(on);
    
    // Plot data points if GP has training data
    if (gp.is_fitted()) {
        auto X_train = gp.getX();
        auto y_train = gp.getY();
        
        if (X_train.cols() == 2) {  // Only plot if 2D
            std::vector<double> x_data, y_data, z_data;
            for (int i = 0; i < X_train.rows(); ++i) {
                x_data.push_back(X_train(i, 0));
                y_data.push_back(X_train(i, 1));
                z_data.push_back(y_train(i));
            }
            
            auto data_plot = scatter3(x_data, y_data, z_data);
            data_plot->marker_size(config.marker_size);
            data_plot->color(config.data_point_color);
            data_plot->color(config.data_point_color);
            data_plot->display_name("Training Data");
            
            // Highlight latest point
            if (!x_data.empty()) {
                std::vector<double> latest_x = {x_data.back()};
                std::vector<double> latest_y = {y_data.back()};
                std::vector<double> latest_z = {z_data.back()};
                auto latest_plot = scatter3(latest_x, latest_y, latest_z);
                latest_plot->marker_size(config.marker_size + 2);
                latest_plot->color(config.latest_point_color);
                latest_plot->color(config.latest_point_color);
                latest_plot->display_name("Latest Point");
            }
        }
    }
    
    xlabel("Input 1");
    ylabel("Input 2");
    zlabel("Output");
    title("SafeOpt 2D Gaussian Process (3D Surface)");
    legend();
    
    show();
}

void SafeOptVisualizer::plot(const SafeOpt& optimizer,
                            const PlotConfig& config) {
    // Get the first GP (objective function)
    auto gps = optimizer.getGPs();
    if (gps.empty()) {
        std::cerr << "Error: No GPs found in optimizer" << std::endl;
        return;
    }
    
    auto gp = gps[0];
    auto bounds = optimizer.getBounds();
    
    // Determine effective dimensionality
    int input_dim = bounds.size();
    int effective_dim = input_dim;
    
    // Handle fixed inputs
    if (!config.fixed_inputs.empty()) {
        effective_dim -= config.fixed_inputs.size();
    }
    
    switch (effective_dim) {
        case 1: {
            std::vector<double> inputs = generate_1d_grid(bounds[0], config.n_samples);
            plot_1d_gp(*gp, inputs, config);
            break;
        }
        case 2: {
            if (config.plot_3d) {
                plot_3d_gp(*gp, bounds, config);
            } else {
                plot_contour_gp(*gp, bounds, config);
            }
            break;
        }
        default:
            std::cerr << "Error: Plotting not supported for dimensions > 2" << std::endl;
            break;
    }
}

void SafeOptVisualizer::plot_history(const SafeOpt& optimizer,
                                    const PlotConfig& config) {
    using namespace matplot;
    
    // Get optimization history
    auto gps = optimizer.getGPs();
    if (gps.empty() || !gps[0]->is_fitted()) {
        std::cerr << "Error: No optimization history available" << std::endl;
        return;
    }
    
    auto gp = gps[0];
    auto y_data = gp->getY();
    
    // Create iteration indices
    std::vector<double> iterations;
    std::vector<double> values;
    for (int i = 0; i < y_data.size(); ++i) {
        iterations.push_back(i + 1);
        values.push_back(y_data(i));
    }
    
    // Calculate cumulative maximum
    std::vector<double> cumulative_max;
    double current_max = values[0];
    for (double val : values) {
        current_max = std::max(current_max, val);
        cumulative_max.push_back(current_max);
    }
    
    figure();
    
    // Plot function values
    auto val_plot = matplot::plot(iterations, values);
    val_plot->color("blue");
    val_plot->marker("o");
    val_plot->line_width(2.0);
    val_plot->display_name("Function Values");
    
    hold(on);
    
    // Plot cumulative maximum
    auto max_plot = matplot::plot(iterations, cumulative_max);
    max_plot->color("red");
    max_plot->line_width(2.0);
    max_plot->display_name("Cumulative Maximum");
    
    // Plot safety threshold if specified
    if (config.show_safety_threshold && !config.fmin.empty()) {
        std::vector<double> threshold_line(iterations.size(), config.fmin[0]);
        auto thresh_plot = matplot::plot(iterations, threshold_line);
        thresh_plot->color("green");
        thresh_plot->line_style("--");
        thresh_plot->line_width(2.0);
        thresh_plot->display_name("Safety Threshold");
    }
    
    xlabel("Iteration");
    ylabel("Function Value");
    title("SafeOpt Optimization History");
    legend();
    grid(on);
    
    show();
}

void SafeOptVisualizer::plot_safe_set(const SafeOpt& optimizer,
                                     const PlotConfig& config) {
    using namespace matplot;
    
    auto bounds = optimizer.getBounds();
    if (bounds.size() != 2) {
        std::cerr << "Error: Safe set visualization only supported for 2D problems" << std::endl;
        return;
    }
    
    // Generate evaluation grid
    auto [x_grid, y_grid] = generate_2d_grid(bounds, config.n_samples);
    Eigen::MatrixXd X_test = grid_to_points(x_grid, y_grid);
    
    // Get safe set and expander information
    auto safe_set = optimizer.getSafeSet();
    auto expanders = optimizer.getExpanders();
    
    // Create figure
    figure();
    
    // Plot safe set points
    std::vector<double> safe_x, safe_y;
    for (int i = 0; i < safe_set.size(); ++i) {
        if (safe_set(i) > 0 && i < X_test.rows()) {
            safe_x.push_back(X_test(i, 0));
            safe_y.push_back(X_test(i, 1));
        }
    }
    
    if (!safe_x.empty()) {
        auto safe_plot = scatter(safe_x, safe_y);
        safe_plot->marker_size(8);
        safe_plot->color("green");
        safe_plot->display_name("Safe Set");
    }
    
    hold(on);
    
    // Plot expander points
    std::vector<double> exp_x, exp_y;
    for (int i = 0; i < expanders.size(); ++i) {
        if (expanders(i) > 0 && i < X_test.rows()) {
            exp_x.push_back(X_test(i, 0));
            exp_y.push_back(X_test(i, 1));
        }
    }
    
    if (!exp_x.empty()) {
        auto exp_plot = scatter(exp_x, exp_y);
        exp_plot->marker_size(10);
        exp_plot->color("orange");
        exp_plot->display_name("Expanders");
    }
    
    // Plot training data
    auto gps = optimizer.getGPs();
    if (!gps.empty() && gps[0]->is_fitted()) {
        auto X_train = gps[0]->getX();
        if (X_train.cols() == 2) {
            std::vector<double> x_data, y_data;
            for (int i = 0; i < X_train.rows(); ++i) {
                x_data.push_back(X_train(i, 0));
                y_data.push_back(X_train(i, 1));
            }
            
            auto data_plot = scatter(x_data, y_data);
            data_plot->marker_size(12);
            data_plot->color("red");
            data_plot->color("red");
            data_plot->display_name("Training Data");
        }
    }
    
    xlabel("Input 1");
    ylabel("Input 2");
    title("SafeOpt Safe Set and Expanders");
    legend();
    grid(on);
    
    show();
}

// Private helper methods

std::vector<double> SafeOptVisualizer::generate_1d_grid(const std::pair<double, double>& bounds,
                                                       int n_samples) const {
    std::vector<double> grid;
    double step = (bounds.second - bounds.first) / (n_samples - 1);
    
    for (int i = 0; i < n_samples; ++i) {
        grid.push_back(bounds.first + i * step);
    }
    
    return grid;
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
SafeOptVisualizer::generate_2d_grid(const std::vector<std::pair<double, double>>& bounds,
                                   int n_samples) const {
    int n_samples_sqrt = static_cast<int>(std::sqrt(n_samples));
    
    std::vector<std::vector<double>> x_grid(n_samples_sqrt, std::vector<double>(n_samples_sqrt));
    std::vector<std::vector<double>> y_grid(n_samples_sqrt, std::vector<double>(n_samples_sqrt));
    
    double x_step = (bounds[0].second - bounds[0].first) / (n_samples_sqrt - 1);
    double y_step = (bounds[1].second - bounds[1].first) / (n_samples_sqrt - 1);
    
    for (int i = 0; i < n_samples_sqrt; ++i) {
        for (int j = 0; j < n_samples_sqrt; ++j) {
            x_grid[i][j] = bounds[0].first + j * x_step;
            y_grid[i][j] = bounds[1].first + i * y_step;
        }
    }
    
    return {x_grid, y_grid};
}

Eigen::MatrixXd SafeOptVisualizer::grid_to_points(const std::vector<std::vector<double>>& x_grid,
                                                  const std::vector<std::vector<double>>& y_grid) const {
    int n_points = x_grid.size() * x_grid[0].size();
    Eigen::MatrixXd points(n_points, 2);
    
    int idx = 0;
    for (size_t i = 0; i < x_grid.size(); ++i) {
        for (size_t j = 0; j < x_grid[i].size(); ++j) {
            points(idx, 0) = x_grid[i][j];
            points(idx, 1) = y_grid[i][j];
            idx++;
        }
    }
    
    return points;
}

void SafeOptVisualizer::plot_safety_threshold(const std::pair<double, double>& x_range,
                                             double fmin) const {
    using namespace matplot;
    
    std::vector<double> x_line = {x_range.first, x_range.second};
    std::vector<double> y_line = {fmin, fmin};
    
    auto thresh_plot = matplot::plot(x_line, y_line);
    thresh_plot->color("red");
    thresh_plot->line_style("--");
    thresh_plot->line_width(2.0);
    thresh_plot->display_name("Safety Threshold");
}

std::pair<std::vector<double>, std::vector<double>> 
SafeOptVisualizer::calculate_confidence_intervals(const std::vector<double>& mean,
                                                 const std::vector<double>& variance,
                                                 double beta) const {
    std::vector<double> lower_bound, upper_bound;
    
    for (size_t i = 0; i < mean.size(); ++i) {
        double std_dev = beta * std::sqrt(variance[i]);
        lower_bound.push_back(mean[i] - std_dev);
        upper_bound.push_back(mean[i] + std_dev);
    }
    
    return {lower_bound, upper_bound};
}

// Utility functions

namespace viz_utils {

Eigen::MatrixXd linearly_spaced_combinations(
    const std::vector<std::pair<double, double>>& bounds,
    int num_samples) {
    
    int n_dims = bounds.size();
    int total_samples = static_cast<int>(std::pow(num_samples, n_dims));
    
    Eigen::MatrixXd combinations(total_samples, n_dims);
    
    for (int dim = 0; dim < n_dims; ++dim) {
        double range = bounds[dim].second - bounds[dim].first;
        double step = range / (num_samples - 1);
        
        int repeat_count = static_cast<int>(std::pow(num_samples, n_dims - dim - 1));
        int cycle_count = total_samples / (num_samples * repeat_count);
        
        for (int i = 0; i < total_samples; ++i) {
            int cycle_idx = (i / repeat_count) % num_samples;
            combinations(i, dim) = bounds[dim].first + cycle_idx * step;
        }
    }
    
    return combinations;
}

Eigen::MatrixXd linearly_spaced_combinations(
    const std::vector<std::pair<double, double>>& bounds,
    const std::vector<int>& num_samples) {
    
    int n_dims = bounds.size();
    if (static_cast<int>(num_samples.size()) != n_dims) {
        throw std::invalid_argument("Number of samples must match number of dimensions");
    }
    
    int total_samples = 1;
    for (int samples : num_samples) {
        total_samples *= samples;
    }
    
    Eigen::MatrixXd combinations(total_samples, n_dims);
    
    for (int dim = 0; dim < n_dims; ++dim) {
        double range = bounds[dim].second - bounds[dim].first;
        double step = range / (num_samples[dim] - 1);
        
        int repeat_count = 1;
        for (int d = dim + 1; d < n_dims; ++d) {
            repeat_count *= num_samples[d];
        }
        
        for (int i = 0; i < total_samples; ++i) {
            int cycle_idx = (i / repeat_count) % num_samples[dim];
            combinations(i, dim) = bounds[dim].first + cycle_idx * step;
        }
    }
    
    return combinations;
}

} // namespace viz_utils

} // namespace safeopt