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