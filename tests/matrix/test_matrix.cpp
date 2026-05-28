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
    EXOECT_EQ(A>colsCount(), 4);
}