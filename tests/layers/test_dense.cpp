#include "neural_engine/layers/Dense.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

TEST(DenseTest, ForwardReturnsCorrectShape) {
    Dense layer(3, 2);

    Matrix input(1, 3);
    input(0, 0) = 2.0;
    input(0, 1) = 4.0;
    input(0, 2) = 6.0;

    Matrix output = layer.forward(input);

    EXPECT_EQ(output.rowsCount(), 1);
    EXPECT_EQ(output.colsCount(), 2);
}

TEST(DenseTest, ForwardThrowsForWrongInputSize) {
    Dense layer(3, 2);

    Matrix input(1, 4);

    EXPECT_THROW(layer.forward(input), std::invalid_argument);
}