cmake_minimum_required(VERSION 3.16)

project(d8_inference LANGUAGES CXX)

add_executable(d8_inference main.cpp)
target_compile_features(d8_inference PRIVATE cxx_std_20)
set_target_properties(d8_inference PROPERTIES CXX_EXTENSIONS OFF)

# Standalone prefill simulator; require standard C++20 without extensions.
add_executable(prefill_sim prefill_sim.cpp)
set_target_properties(prefill_sim PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(d8_inference PRIVATE -Wall -Wextra -Wpedantic)
    # Keep frame pointers for readable Linux perf call stacks.
    target_compile_options(prefill_sim PRIVATE
        -Wall -Wextra -Wpedantic -fno-omit-frame-pointer
    )
endif()

# Standalone decode simulator using standard C++20.
add_executable(decode_sim decode_sim.cpp)
set_target_properties(decode_sim PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(decode_sim PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_custom_target(
    run
    COMMAND $<TARGET_FILE:d8_inference>
    DEPENDS d8_inference
    USES_TERMINAL
)
