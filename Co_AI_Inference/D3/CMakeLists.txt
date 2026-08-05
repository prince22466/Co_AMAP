cmake_minimum_required(VERSION 3.16)
project(llm_fifo_percentile_simulator LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(llm_fifo_percentile_simulator main.cpp)
target_compile_features(llm_fifo_percentile_simulator PRIVATE cxx_std_20)
set_target_properties(
    llm_fifo_percentile_simulator
    PROPERTIES
        CXX_EXTENSIONS OFF
)
target_link_libraries(
    llm_fifo_percentile_simulator
    PRIVATE
        Threads::Threads
)
