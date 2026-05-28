#include "neural_engine/matrix/Matrix.hpp"
#include <stdexcept>
#include <gtest/gtest.h>

//Using the equal function of google test we can see if it fills with zeroes without actually seeing the matrix
TEST(MatrixTest, ConstructorCreatesZeroFilledMatrix) {
    //Matrix Object created with 2 rows and 3 columns
    Matrix A(2,3); //Whenver it is created, it should be filled with zeroes
    EXPECT_DOUBLE_EQ(A(0,0),0.0);
    EXPECT_DOUBLE_EQ(A(1,2),0.0);
    EXPECT_DOUBLE_EQ(A(0,2),0.0);

    
}

TEST(MatrixTest, CanStoreAndReadValues) {
    // Create Object A with 2 rows and 3 columns
    Matrix A(2,3);

    // This allows to access the index
    // Doing this will take to first row 1, first column 1
    A(0,0) = 5.0;
    A(1,2) = 10.0;

    //This is from GoogleTest, it checks if the output matches the expected output
    EXPECT_DOUBLE_EQ(A(0,0), 5.0);
    EXPECT_DOUBLE_EQ(A(1,2), 10.0);
}

// This should thrown an error if index is outside the range
TEST(MatrixTest, ThrowsForInvalidIndex) {

    // test 3
    Matrix A(2, 3);

    EXPECT_THROW(A(2,0), std::out_of_range);
    EXPECT_THROW(A(0,3), std::out_of_range);
    EXPECT_THROW(A(-1,0), std::out_of_range);


}

TEST(MatrixTest, ReturnsDimensions) {
    Matrix A(2,4);

    EXPECT_EQ(A.rowsCount(), 2);
    EXPECT_EQ(A.colsCount(), 4);
}
TEST(MatrixTest, CanAddMatrices) {
    Matrix A(2, 2);
    Matrix B(2, 2);

    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;

    B(0, 0) = 5;
    B(0, 1) = 6;
    B(1, 0) = 7;
    B(1, 1) = 8;

    Matrix C = A.matrixAddition(B);

    EXPECT_DOUBLE_EQ(C(0, 0), 6);
    EXPECT_DOUBLE_EQ(C(0, 1), 8);
    EXPECT_DOUBLE_EQ(C(1, 0), 10);
    EXPECT_DOUBLE_EQ(C(1, 1), 12);
}

TEST(MatrixTest, AdditionThrowsForDifferentDimensions) {
    Matrix A(2, 3);
    Matrix B(3, 2);

    EXPECT_THROW(A.matrixAddition(B), std::invalid_argument);
}

TEST(MatrixTest, CanSubtractMatrices) {
    Matrix A(2, 2);
    Matrix B(2, 2);

    A(0, 0) = 5;
    A(0, 1) = 7;
    A(1, 0) = 9;
    A(1, 1) = 11;

    B(0, 0) = 1;
    B(0, 1) = 2;
    B(1, 0) = 3;
    B(1, 1) = 4;

    Matrix C = A.matrixSubtraction(B);

    EXPECT_DOUBLE_EQ(C(0, 0), 4);
    EXPECT_DOUBLE_EQ(C(0, 1), 5);
    EXPECT_DOUBLE_EQ(C(1, 0), 6);
    EXPECT_DOUBLE_EQ(C(1, 1), 7);
}

TEST(MatrixTest, SubtractionThrowsForDifferentDimensions) {
    Matrix A(2, 3);
    Matrix B(3, 2);

    EXPECT_THROW(A.matrixSubtraction(B), std::invalid_argument);
}

TEST(MatrixTest, CanMultiplyByScalar) {
    Matrix A(2, 2);

    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;

    Matrix B = A.multiplyByScalar(3.0);

    EXPECT_DOUBLE_EQ(B(0, 0), 3);
    EXPECT_DOUBLE_EQ(B(0, 1), 6);
    EXPECT_DOUBLE_EQ(B(1, 0), 9);
    EXPECT_DOUBLE_EQ(B(1, 1), 12);
}


TEST(MatrixTest, CanMultiplyMatrices) {
    Matrix A(2, 3);
    Matrix B(3, 2);

    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;

    B(0, 0) = 7;
    B(0, 1) = 8;
    B(1, 0) = 9;
    B(1, 1) = 10;
    B(2, 0) = 11;
    B(2, 1) = 12;

    Matrix C = A.matrixMultiplication(B);

    EXPECT_DOUBLE_EQ(C(0, 0), 58);
    EXPECT_DOUBLE_EQ(C(0, 1), 64);
    EXPECT_DOUBLE_EQ(C(1, 0), 139);
    EXPECT_DOUBLE_EQ(C(1, 1), 154);
}

TEST(MatrixTest, MultiplicationThrowsForInvalidDimensions) {
    Matrix A(2, 3);
    Matrix B(2, 2);

    EXPECT_THROW(A.matrixMultiplication(B), std::invalid_argument);
}