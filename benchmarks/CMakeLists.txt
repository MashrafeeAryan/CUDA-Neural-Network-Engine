add_executable(benchmark_cpu_matrix_mult
    benchmark_cpu_matrix_mult.cpp
)

target_link_libraries(benchmark_cpu_matrix_mult
    PRIVATE
        matrix
)