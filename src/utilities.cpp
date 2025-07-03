#include "safeopt/utilities.hpp"
#include <algorithm>
#include <random>
#include <cmath>

namespace safeopt {

Eigen::MatrixXd linearly_spaced_combinations(
    const std::vector<std::pair<double, double>>& bounds,
    const std::vector<int>& num_samples) {
    
    if (bounds.size() != num_samples.size()) {
        throw std::invalid_argument("bounds and num_samples must have same size");
    }

    int ndim = bounds.size();
    
    // Calculate total number of combinations
    int total_samples = 1;
    for (int ns : num_samples) {
        total_samples *= ns;
    }
    
    Eigen::MatrixXd result(total_samples, ndim);
    
    // Generate linearly spaced points for each dimension
    std::vector<Eigen::VectorXd> linspaces(ndim);
    for (int d = 0; d < ndim; ++d) {
        linspaces[d] = Eigen::VectorXd::LinSpaced(num_samples[d], 
                                                 bounds[d].first, 
                                                 bounds[d].second);
    }
    
    // Generate all combinations
    int row = 0;
    std::function<void(int, Eigen::VectorXd&)> generate_combinations = 
        [&](int dim, Eigen::VectorXd& current) {
            if (dim == ndim) {
                result.row(row++) = current;
                return;
            }
            
            for (int i = 0; i < num_samples[dim]; ++i) {
                current[dim] = linspaces[dim][i];
                generate_combinations(dim + 1, current);
            }
        };
    
    Eigen::VectorXd current(ndim);
    generate_combinations(0, current);
    
    return result;
}

Eigen::MatrixXd linearly_spaced_combinations(
    const std::vector<std::pair<double, double>>& bounds,
    int num_samples) {
    
    std::vector<int> samples_per_dim(bounds.size(), num_samples);
    return linearly_spaced_combinations(bounds, samples_per_dim);
}

std::function<Eigen::VectorXd(const Eigen::MatrixXd&, bool)> 
sample_gp_function(
    const std::vector<std::pair<double, double>>& bounds,
    double noise_var,
    int num_samples) {
    
    // Generate discretization points
    Eigen::MatrixXd inputs = linearly_spaced_combinations(bounds, num_samples);
    int n_points = inputs.rows();
    int ndim = inputs.cols();
    
    // Create a simple RBF kernel covariance matrix
    Eigen::MatrixXd cov(n_points, n_points);
    double lengthscale = 1.0;  // Fixed lengthscale
    double variance = 1.0;     // Fixed variance
    
    for (int i = 0; i < n_points; ++i) {
        for (int j = 0; j < n_points; ++j) {
            double dist_sq = (inputs.row(i) - inputs.row(j)).squaredNorm();
            cov(i, j) = variance * std::exp(-0.5 * dist_sq / (lengthscale * lengthscale));
        }
    }
    
    // Add small diagonal term for numerical stability
    cov.diagonal().array() += 1e-6;
    
    // Sample from multivariate normal
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> norm_dist(0.0, 1.0);
    
    Eigen::VectorXd z(n_points);
    for (int i = 0; i < n_points; ++i) {
        z[i] = norm_dist(gen);
    }
    
    // Cholesky decomposition for sampling
    Eigen::LLT<Eigen::MatrixXd> llt(cov);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("Failed to compute Cholesky decomposition");
    }
    
    Eigen::VectorXd output = llt.matrixL() * z;
    
    // Return evaluation function using linear interpolation
    return [inputs, output, noise_var](const Eigen::MatrixXd& x, bool add_noise) -> Eigen::VectorXd {
        int n_query = x.rows();
        Eigen::VectorXd result(n_query);
        
        // Simple nearest neighbor interpolation for now
        // In a real implementation, you might want to use proper interpolation
        for (int i = 0; i < n_query; ++i) {
            double min_dist = std::numeric_limits<double>::infinity();
            int nearest_idx = 0;
            
            for (int j = 0; j < inputs.rows(); ++j) {
                double dist = (x.row(i) - inputs.row(j)).norm();
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_idx = j;
                }
            }
            
            result[i] = output[nearest_idx];
            
            if (add_noise) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::normal_distribution<double> noise_dist(0.0, std::sqrt(noise_var));
                result[i] += noise_dist(gen);
            }
        }
        
        return result;
    };
}

} // namespace safeopt