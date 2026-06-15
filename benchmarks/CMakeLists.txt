add_executable(benchmark_cpu_matrix_mult
    benchmark_cpu_matrix_mult.cpp
)

target_link_libraries(benchmark_cpu_matrix_mult
    PRIVATE
        matrix
)

if(ENABLE_CUDA)
    add_executable(benchmark_cuda_matrix_mult
        benchmark_cuda_matrix_mult.cu
    )

    target_link_libraries(benchmark_cuda_matrix_mult
        PRIVATE
            cuda_matrix_mult
            matrix
    )

    add_executable(benchmark_matrix_cpu_vs_cuda
        benchmark_matrix_cpu_vs_cuda.cu
    )

    target_link_libraries(benchmark_matrix_cpu_vs_cuda
        PRIVATE
            cuda_matrix_mult
            matrix
    )
endif()

