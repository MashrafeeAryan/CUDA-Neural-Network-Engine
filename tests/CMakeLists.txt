#Module in CMake that allows CMake to browse internet and get neccessary files
include (FetchContent)

#It shows what we want to download and then we give it a link to download from this place
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

# Helps avoid runtime library issues on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)


#Triggers CMake to go and download the zip file and add it the build
FetchContent_MakeAvailable(googletest)


#Compiles test_matrix.cpp and converts it into executable
#matrix_tests is an executable name
add_executable(matrix_tests
    matrix/test_matrix.cpp
)

add_executable(dense_tests
    layers/test_dense.cpp
)
add_executable(relu_tests
    activations/test_relu.cpp
)

add_executable(sigmoid_tests
activations/test_sigmoid.cpp
)

add_executable(softmax_tests
activations/test_softmax.cpp)


add_executable(mse_loss_tests
loss/test_mse_loss.cpp)

target_link_libraries(matrix_tests
    PRIVATE
        matrix
        GTest::gtest_main
)

target_link_libraries(dense_tests
    PRIVATE
        matrix
        layers
        GTest::gtest_main
)

target_link_libraries(relu_tests
    PRIVATE
        matrix
        activations

        GTest::gtest_main
)

target_link_libraries(sigmoid_tests
    PRIVATE
        matrix
        activations

        GTest::gtest_main
)

target_link_libraries(softmax_tests
    PRIVATE
        matrix
        activations

        GTest::gtest_main
)

target_link_libraries(mse_loss_tests
    PRIVATE
        matrix
        loss

        GTest::gtest_main
)
# Registers the GoogleTest tests with CTest
include(GoogleTest)
gtest_discover_tests(matrix_tests)
gtest_discover_tests(dense_tests)
gtest_discover_tests(relu_tests)
gtest_discover_tests(sigmoid_tests)
gtest_discover_tests(softmax_tests)
gtest_discover_tests(mse_loss_tests)