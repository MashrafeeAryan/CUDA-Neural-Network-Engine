add_executable(neural_engine_app
    main.cpp
)

target_link_libraries(neural_engine_app
    PRIVATE
        matrix
        layers
        activations
)