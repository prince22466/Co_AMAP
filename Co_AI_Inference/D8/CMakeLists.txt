cmake_minimum_required(VERSION 3.16)

project(d8_inference LANGUAGES CXX)

add_executable(d8_inference main.cpp)
target_compile_features(d8_inference PRIVATE cxx_std_20)
set_target_properties(d8_inference PROPERTIES CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(d8_inference PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_custom_target(
    run
    COMMAND $<TARGET_FILE:d8_inference>
    DEPENDS d8_inference
    USES_TERMINAL
)