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

TEST(DenseTest, ForwardSupportsBatchInput) {
    Dense layer(3, 2);

    Matrix input(4, 3);

    Matrix output = layer.forward(input);

    EXPECT_EQ(output.rowsCount(), 4);
    EXPECT_EQ(output.colsCount(), 2);
}


TEST(DenseTest, BackwardReturnsInputGradientWithCorrectShape) {
    Dense layer(3, 2);

    Matrix input(4, 3);
    layer.forward(input);

    Matrix outputGradient(4, 2);

    Matrix inputGradient = layer.backward(outputGradient);

    EXPECT_EQ(inputGradient.rowsCount(), 4);
    EXPECT_EQ(inputGradient.colsCount(), 3);
}

TEST(DenseTest, BackwardStoresWeightAndBiasGradientsWithCorrectShape) {
    Dense layer(3, 2);

    Matrix input(4, 3);
    layer.forward(input);

    Matrix outputGradient(4, 2);
    layer.backward(outputGradient);

    Matrix weightGradients = layer.getWeightGradients();
    Matrix biasGradients = layer.getBiasGradients();

    EXPECT_EQ(weightGradients.rowsCount(), 3);
    EXPECT_EQ(weightGradients.colsCount(), 2);

    EXPECT_EQ(biasGradients.rowsCount(), 1);
    EXPECT_EQ(biasGradients.colsCount(), 2);
}

TEST(DenseTest, BackwardCalculatesBiasGradients) {
    Dense layer(3, 2);

    Matrix input(2, 3);
    layer.forward(input);

    Matrix outputGradient(2, 2);
    outputGradient(0, 0) = 1.0;
    outputGradient(0, 1) = 2.0;
    outputGradient(1, 0) = 3.0;
    outputGradient(1, 1) = 4.0;

    layer.backward(outputGradient);

    Matrix biasGradients = layer.getBiasGradients();

    EXPECT_NEAR(biasGradients(0, 0), 4.0, 1e-6); // 1 + 3
    EXPECT_NEAR(biasGradients(0, 1), 6.0, 1e-6); // 2 + 4
}

TEST(DenseTest, BackwardThrowsForWrongOutputGradientShape) {
    Dense layer(3, 2);

    Matrix input(4, 3);
    layer.forward(input);

    Matrix wrongOutputGradient(4, 3);

    EXPECT_THROW(layer.backward(wrongOutputGradient), std::invalid_argument);
}