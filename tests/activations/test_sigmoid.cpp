#include "neural_engine/activations/Sigmoid.hpp"

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

