add_library(cuda_matrix_mult
    CUDA_MatrixMult.cu
)

target_include_directories(cuda_matrix_mult
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(cuda_matrix_mult
    PUBLIC
        matrix
)