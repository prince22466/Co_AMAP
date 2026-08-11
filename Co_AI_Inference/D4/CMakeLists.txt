cmake_minimum_required(VERSION 3.16)
project(llm_fifo_simulator LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(fifo_sim main.cpp)
target_compile_features(fifo_sim PRIVATE cxx_std_20)
set_target_properties(fifo_sim PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(fifo_sim PRIVATE Threads::Threads)

# Preserve readable call stacks when profiling optimized builds with perf.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(fifo_sim PRIVATE -fno-omit-frame-pointer)
endif()
