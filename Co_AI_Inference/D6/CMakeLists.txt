cmake_minimum_required(VERSION 3.16)

project(request_layout_benchmark LANGUAGES CXX)

add_executable(request_layout_benchmark main_1.cpp)
target_compile_features(request_layout_benchmark PRIVATE cxx_std_17)
set_target_properties(request_layout_benchmark PROPERTIES CXX_EXTENSIONS OFF)

add_executable(aos_soa_benchmark main_2.cpp)
target_compile_features(aos_soa_benchmark PRIVATE cxx_std_17)
set_target_properties(aos_soa_benchmark PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(request_layout_benchmark PRIVATE /W4)
    target_compile_options(aos_soa_benchmark PRIVATE /W4)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        request_layout_benchmark
        PRIVATE -Wall -Wextra -Wpedantic -fno-omit-frame-pointer
    )
    target_compile_options(
        aos_soa_benchmark
        PRIVATE -Wall -Wextra -Wpedantic -fno-omit-frame-pointer
    )
endif()

if(WIN32)
    target_link_libraries(request_layout_benchmark PRIVATE psapi)
endif()

add_custom_target(
    run_compact
    COMMAND $<TARGET_FILE:request_layout_benchmark> --mode compact
    DEPENDS request_layout_benchmark
    USES_TERMINAL
)

add_custom_target(
    run_large
    COMMAND $<TARGET_FILE:request_layout_benchmark> --mode large
    DEPENDS request_layout_benchmark
    USES_TERMINAL
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_program(PERF_EXECUTABLE perf)
    if(PERF_EXECUTABLE)
        set(PERF_EVENTS
            "task-clock,cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses"
        )

        add_custom_target(
            perf_compact
            COMMAND ${PERF_EXECUTABLE} stat -r 5 -e ${PERF_EVENTS} --
                    $<TARGET_FILE:request_layout_benchmark> --mode compact
                    --iterations 100
            DEPENDS request_layout_benchmark
            USES_TERMINAL
        )

        add_custom_target(
            perf_large
            COMMAND ${PERF_EXECUTABLE} stat -r 5 -e ${PERF_EVENTS} --
                    $<TARGET_FILE:request_layout_benchmark> --mode large
                    --iterations 100
            DEPENDS request_layout_benchmark
            USES_TERMINAL
        )

        add_custom_target(
            perf_aos
            COMMAND ${PERF_EXECUTABLE} stat -r 5 -e ${PERF_EVENTS} --
                    $<TARGET_FILE:aos_soa_benchmark> --mode aos
            DEPENDS aos_soa_benchmark
            USES_TERMINAL
        )

        add_custom_target(
            perf_soa
            COMMAND ${PERF_EXECUTABLE} stat -r 5 -e ${PERF_EVENTS} --
                    $<TARGET_FILE:aos_soa_benchmark> --mode soa
            DEPENDS aos_soa_benchmark
            USES_TERMINAL
        )
    endif()
endif()
