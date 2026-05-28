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

target_link_libraries(matrix_tests
    PRIVATE
        matrix
        GTest::gtest_main
)


# Registers the GoogleTest tests with CTest
include(GoogleTest)
gtest_discover_tests(matrix_tests)