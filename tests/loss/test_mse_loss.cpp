#include "neural_engine/loss/MSE_Loss.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

TEST(MSELossTest, ForwardReturnsZeroWhenPredictionMatchesTarget) {
    MSE_Loss loss;

    Matrix predictions(1, 2);
    predictions(0, 0) = 1.0;
    predictions(0, 1) = 2.0;

    Matrix targets(1, 2);
    targets(0, 0) = 1.0;
    targets(0, 1) = 2.0;

    double result = loss.forward(predictions, targets);

    EXPECT_NEAR(result, 0.0, 1e-6);
}

TEST(MSELossTest, ForwardCalculatesMeanSquaredError) {
    MSE_Loss loss;

    Matrix predictions(1, 2);
    predictions(0, 0) = 0.0;
    predictions(0, 1) = 2.0;

    Matrix targets(1, 2);
    targets(0, 0) = 1.0;
    targets(0, 1) = 4.0;

    double result = loss.forward(predictions, targets);

    // Differences:
    // 0 - 1 = -1, squared = 1
    // 2 - 4 = -2, squared = 4
    // MSE = (1 + 4) / 2 = 2.5
    EXPECT_NEAR(result, 2.5, 1e-6);
}

TEST(MSELossTest, ForwardThrowsForShapeMismatch) {
    MSE_Loss loss;

    Matrix predictions(1, 2);
    Matrix targets(2, 2);

    EXPECT_THROW(loss.forward(predictions, targets), std::invalid_argument);
}

TEST(MSELossTest, BackwardCalculatesGradient) {
    MSE_Loss loss;

    Matrix predictions(1, 2);
    predictions(0, 0) = 0.8;
    predictions(0, 1) = 0.2;

    Matrix targets(1, 2);
    targets(0, 0) = 1.0;
    targets(0, 1) = 0.0;

    Matrix gradient = loss.backward(predictions, targets);

    // Formula: 2 * (prediction - target) / number_of_values
    // number_of_values = 2
    // gradient[0] = 2 * (0.8 - 1.0) / 2 = -0.2
    // gradient[1] = 2 * (0.2 - 0.0) / 2 = 0.2

    EXPECT_NEAR(gradient(0, 0), -0.2, 1e-6);
    EXPECT_NEAR(gradient(0, 1), 0.2, 1e-6);
}

TEST(MSELossTest, BackwardReturnsSameShapeAsPredictions) {
    MSE_Loss loss;

    Matrix predictions(2, 3);
    Matrix targets(2, 3);

    Matrix gradient = loss.backward(predictions, targets);

    EXPECT_EQ(gradient.rowsCount(), 2);
    EXPECT_EQ(gradient.colsCount(), 3);
}

TEST(MSELossTest, BackwardThrowsForShapeMismatch) {
    MSE_Loss loss;

    Matrix predictions(1, 2);
    Matrix targets(2, 2);

    EXPECT_THROW(loss.backward(predictions, targets), std::invalid_argument);
}