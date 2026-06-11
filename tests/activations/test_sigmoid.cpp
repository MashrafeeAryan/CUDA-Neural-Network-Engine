#include "neural_engine/activations/Sigmoid.hpp"
#include <stdexcept>
#include <gtest/gtest.h>

TEST(SigmoidTest, ForwardConvertsZeroToPointFive) {
    Sigmoid sigmoid;

    Matrix input(1, 1);
    input(0, 0) = 0.0;

    Matrix output = sigmoid.forward(input);

    EXPECT_NEAR(output(0, 0), 0.5, 1e-6);
}

TEST(SigmoidTest, ForwardKeepsSameShape) {
    Sigmoid sigmoid;

    Matrix input(3, 4);

    Matrix output = sigmoid.forward(input);

    EXPECT_EQ(output.rowsCount(), 3);
    EXPECT_EQ(output.colsCount(), 4);
}

TEST(SigmoidTest, ForwardOutputsValuesBetweenZeroAndOne) {
    Sigmoid sigmoid;

    Matrix input(1, 3);
    input(0, 0) = -10.0;
    input(0, 1) = 0.0;
    input(0, 2) = 10.0;

    Matrix output = sigmoid.forward(input);

    EXPECT_GT(output(0, 0), 0.0);
    EXPECT_LT(output(0, 0), 1.0);

    EXPECT_GT(output(0, 1), 0.0);
    EXPECT_LT(output(0, 1), 1.0);

    EXPECT_GT(output(0, 2), 0.0);
    EXPECT_LT(output(0, 2), 1.0);
}

TEST(SigmoidTest, BackwardCalculatesInputGradient) {
    Sigmoid sigmoid;

    Matrix input(1, 2);
    input(0, 0) = 0.0;
    input(0, 1) = 0.0;

    sigmoid.forward(input);

    Matrix outputGradient(1, 2);
    outputGradient(0, 0) = 10.0;
    outputGradient(0, 1) = 20.0;

    Matrix inputGradient = sigmoid.backward(outputGradient);

    // sigmoid(0) = 0.5
    // derivative = 0.5 * (1 - 0.5) = 0.25
    // inputGradient = outputGradient * 0.25
    EXPECT_NEAR(inputGradient(0, 0), 2.5, 1e-6);
    EXPECT_NEAR(inputGradient(0, 1), 5.0, 1e-6);
}

TEST(SigmoidTest, BackwardReturnsSameShapeAsOutputGradient) {
    Sigmoid sigmoid;

    Matrix input(3, 4);
    sigmoid.forward(input);

    Matrix outputGradient(3, 4);

    Matrix inputGradient = sigmoid.backward(outputGradient);

    EXPECT_EQ(inputGradient.rowsCount(), 3);
    EXPECT_EQ(inputGradient.colsCount(), 4);
}

TEST(SigmoidTest, BackwardThrowsForWrongGradientShape) {
    Sigmoid sigmoid;

    Matrix input(2, 2);
    sigmoid.forward(input);

    Matrix wrongOutputGradient(2, 3);

    EXPECT_THROW(sigmoid.backward(wrongOutputGradient), std::invalid_argument);
}