#include "neural_engine/activations/ReLU.hpp"

#include <gtest/gtest.h>

TEST(ReLUTest, ForwardTurnsNegativeValuesToZero) {
    ReLU relu;

    Matrix input(2, 2);
    input(0, 0) = -2.0;
    input(0, 1) = 3.0;
    input(1, 0) = 0.0;
    input(1, 1) = -5.0;

    Matrix output = relu.forward(input);

    EXPECT_EQ(output(0, 0), 0.0);
    EXPECT_EQ(output(0, 1), 3.0);
    EXPECT_EQ(output(1, 0), 0.0);
    EXPECT_EQ(output(1, 1), 0.0);
}
TEST(ReLUTest, ForwardKeepsSameShape) {
    ReLU relu;

    Matrix input(3, 4);

    Matrix output = relu.forward(input);

    EXPECT_EQ(output.rowsCount(), 3);
    EXPECT_EQ(output.colsCount(), 4);
}