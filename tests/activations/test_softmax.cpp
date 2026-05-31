#include "neural_engine/activations/Softmax.hpp"

#include <gtest/gtest.h>

TEST(SoftmaxTest, ForwardOutputsValuesThatSumToOneForSingleRow) {
    Softmax softmax;

    Matrix input(1, 3);
    input(0, 0) = 2.0;
    input(0, 1) = 1.0;
    input(0, 2) = 0.1;

    Matrix output = softmax.forward(input);

    double rowSum = output(0, 0) + output(0, 1) + output(0, 2);

    EXPECT_NEAR(rowSum, 1.0, 1e-6);
}

TEST(SoftmaxTest, ForwardKeepsSameShape) {
    Softmax softmax;

    Matrix input(2, 3);

    Matrix output = softmax.forward(input);

    EXPECT_EQ(output.rowsCount(), 2);
    EXPECT_EQ(output.colsCount(), 3);
}

TEST(SoftmaxTest, ForwardLargestInputGetsLargestProbability) {
    Softmax softmax;

    Matrix input(1, 3);
    input(0, 0) = 1.0;
    input(0, 1) = 5.0;
    input(0, 2) = 2.0;

    Matrix output = softmax.forward(input);

    EXPECT_GT(output(0, 1), output(0, 0));
    EXPECT_GT(output(0, 1), output(0, 2));
}