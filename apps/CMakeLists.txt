add_executable(neural_engine_app
    main.cpp
)

add_executable(train_simple
    train_simple.cpp
)

target_link_libraries(neural_engine_app
    PRIVATE
        matrix
        layers
        activations
)

target_link_libraries(train_simple
    PRIVATE
        matrix
        layers
        loss
)