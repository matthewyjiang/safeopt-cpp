#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <memory>
#include <cmath>
#include "gaussian_process.h"
#include "rbf_kernel.h"
#include "safeopt/gaussian_process_optimization.hpp"

class GPLibraryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test data
        x_train.resize(5, 1);
        x_train << 0.0, 1.0, 2.0, 3.0, 4.0;
        
        y_train.resize(5);
        y_train << 0.0, 1.0, 4.0, 9.0, 16.0;  // y = x^2
        
        x_test.resize(3, 1);
        x_test << 0.5, 1.5, 2.5;
    }
    
    Eigen::MatrixXd x_train, x_test;
    Eigen::VectorXd y_train;
};

TEST_F(GPLibraryTest, BasicGPFunctionality) {
    // Create RBF kernel
    auto kernel = std::make_unique<gp::RBFKernel>(1.0, 1.0);
    
    // Create GP
    gp::GaussianProcess gp(std::move(kernel), 1e-6);
    
    // Test that GP is not fitted initially
    EXPECT_FALSE(gp.is_fitted());
    
    // Fit GP
    gp.fit(x_train, y_train);
    
    // Test that GP is fitted
    EXPECT_TRUE(gp.is_fitted());
    
    // Make predictions
    auto predictions = gp.predict(x_test, true);
    Eigen::VectorXd y_pred = predictions.first;
    Eigen::VectorXd y_std = predictions.second;
    
    // Basic sanity checks
    EXPECT_EQ(y_pred.size(), 3);
    EXPECT_EQ(y_std.size(), 3);
    
    // Check that predictions are reasonable (between 0 and 16)
    for (int i = 0; i < y_pred.size(); ++i) {
        EXPECT_GE(y_pred[i], -5.0);
        EXPECT_LE(y_pred[i], 20.0);
        EXPECT_GT(y_std[i], 0.0);  // Standard deviation should be positive
    }
}

TEST_F(GPLibraryTest, IncrementalLearning) {
    // Create RBF kernel
    auto kernel = std::make_unique<gp::RBFKernel>(1.0, 1.0);
    
    // Create GP
    gp::GaussianProcess gp(std::move(kernel), 1e-6);
    
    // Fit with initial data
    Eigen::MatrixXd x_initial = x_train.topRows(3);
    Eigen::VectorXd y_initial = y_train.head(3);
    gp.fit(x_initial, y_initial);
    
    // Add new data points
    Eigen::VectorXd x_new(1);
    x_new << 3.0;
    gp.add_data_point(x_new, 9.0);
    
    x_new << 4.0;
    gp.add_data_point(x_new, 16.0);
    
    // Make predictions
    auto predictions = gp.predict(x_test, true);
    Eigen::VectorXd y_pred = predictions.first;
    Eigen::VectorXd y_std = predictions.second;
    
    // Check that predictions are reasonable
    EXPECT_EQ(y_pred.size(), 3);
    EXPECT_EQ(y_std.size(), 3);
    
    for (int i = 0; i < y_pred.size(); ++i) {
        EXPECT_GT(y_std[i], 0.0);
    }
}

TEST_F(GPLibraryTest, LogMarginalLikelihood) {
    // Create RBF kernel
    auto kernel = std::make_unique<gp::RBFKernel>(1.0, 1.0);
    
    // Create GP
    gp::GaussianProcess gp(std::move(kernel), 1e-6);
    
    // Fit GP
    gp.fit(x_train, y_train);
    
    // Compute log marginal likelihood
    double log_likelihood = gp.log_marginal_likelihood();
    
    // Should be a finite value
    EXPECT_TRUE(std::isfinite(log_likelihood));
    
    // For well-fitted data, log likelihood should not be extremely negative
    EXPECT_GT(log_likelihood, -1000.0);
}

TEST_F(GPLibraryTest, KernelParameters) {
    // Create RBF kernel with specific parameters
    auto kernel = std::make_unique<gp::RBFKernel>(2.0, 1.5);
    
    // Check parameters before creating GP
    EXPECT_DOUBLE_EQ(kernel->variance(), 2.0);
    EXPECT_DOUBLE_EQ(kernel->lengthscale(), 1.5);
    
    // Test parameter modification
    kernel->set_variance(3.0);
    kernel->set_lengthscale(2.0);
    
    EXPECT_DOUBLE_EQ(kernel->variance(), 3.0);
    EXPECT_DOUBLE_EQ(kernel->lengthscale(), 2.0);
    
    // Create GP and test it still works
    gp::GaussianProcess gp(std::move(kernel), 1e-6);
    gp.fit(x_train, y_train);
    
    auto predictions = gp.predict(x_test, true);
    EXPECT_EQ(predictions.first.size(), 3);
    EXPECT_EQ(predictions.second.size(), 3);
}

TEST_F(GPLibraryTest, SafeOptIntegration) {
    // Create multiple GPs for SafeOpt integration
    std::vector<std::shared_ptr<gp::GaussianProcess>> gps;
    
    // Objective function GP
    auto kernel1 = std::make_unique<gp::RBFKernel>(1.0, 1.0);
    auto gp1 = std::make_shared<gp::GaussianProcess>(std::move(kernel1), 1e-6);
    gps.push_back(gp1);
    
    // Constraint function GP
    auto kernel2 = std::make_unique<gp::RBFKernel>(1.0, 1.0);
    auto gp2 = std::make_shared<gp::GaussianProcess>(std::move(kernel2), 1e-6);
    gps.push_back(gp2);
    
    // Create SafeOpt wrapper
    std::vector<double> fmin = {-10.0, 0.0};  // Objective can be negative, constraint must be positive
    safeopt::GaussianProcessOptimization gpo(gps, fmin, 2.0);
    
    // Add training data
    Eigen::VectorXd x_point(1);
    Eigen::VectorXd y_values(2);
    
    x_point << 1.0;
    y_values << 1.0, 1.0;  // Objective = 1.0, constraint = 1.0
    gpo.addNewDataPoint(x_point, y_values);
    
    x_point << 2.0;
    y_values << 4.0, 0.5;  // Objective = 4.0, constraint = 0.5
    gpo.addNewDataPoint(x_point, y_values);
    
    // Check that data was added correctly
    EXPECT_EQ(gpo.getT(), 2);
    
    auto data = gpo.getData();
    EXPECT_EQ(data.first.rows(), 2);
    EXPECT_EQ(data.second.rows(), 2);
    EXPECT_EQ(data.second.cols(), 2);
}

TEST_F(GPLibraryTest, EdgeCases) {
    // Test with single data point
    auto kernel = std::make_unique<gp::RBFKernel>(1.0, 1.0);
    gp::GaussianProcess gp(std::move(kernel), 1e-6);
    
    Eigen::MatrixXd x_single(1, 1);
    x_single << 1.0;
    Eigen::VectorXd y_single(1);
    y_single << 2.0;
    
    gp.fit(x_single, y_single);
    
    // Make prediction
    auto predictions = gp.predict(x_single, true);
    
    // Should predict close to the training point
    EXPECT_NEAR(predictions.first[0], 2.0, 1e-2);
    EXPECT_GT(predictions.second[0], 0.0);
}

TEST_F(GPLibraryTest, MultiDimensionalInput) {
    // Test with 2D input
    Eigen::MatrixXd x_2d(4, 2);
    x_2d << 0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0;
    
    Eigen::VectorXd y_2d(4);
    y_2d << 0.0, 1.0, 1.0, 2.0;  // Sum of coordinates
    
    auto kernel = std::make_unique<gp::RBFKernel>(1.0, 1.0);
    gp::GaussianProcess gp(std::move(kernel), 1e-6);
    
    gp.fit(x_2d, y_2d);
    
    // Test prediction
    Eigen::MatrixXd x_test_2d(1, 2);
    x_test_2d << 0.5, 0.5;
    
    auto predictions = gp.predict(x_test_2d, true);
    
    EXPECT_EQ(predictions.first.size(), 1);
    EXPECT_EQ(predictions.second.size(), 1);
    
    // Prediction should be reasonable (around 1.0)
    EXPECT_GT(predictions.first[0], 0.0);
    EXPECT_LT(predictions.first[0], 2.0);
}

// Test main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}