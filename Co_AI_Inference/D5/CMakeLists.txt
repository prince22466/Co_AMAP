cmake_minimum_required(VERSION 3.16)

project(llm_no_scheduler_benchmark LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(llm_no_scheduler main_noScheduler.cpp)
target_compile_features(llm_no_scheduler PRIVATE cxx_std_20)
set_target_properties(llm_no_scheduler PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(llm_no_scheduler PRIVATE Threads::Threads)

# Keep stack traces readable when profiling optimized Linux builds with perf.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(llm_no_scheduler PRIVATE -fno-omit-frame-pointer)
endif()

add_custom_target(
    run_all
    COMMAND $<TARGET_FILE:llm_no_scheduler>
    DEPENDS llm_no_scheduler
    USES_TERMINAL
)

foreach(worker_count IN ITEMS 2 4 8 16)
    add_custom_target(
        run_${worker_count}
        COMMAND $<TARGET_FILE:llm_no_scheduler> --workers ${worker_count}
        DEPENDS llm_no_scheduler
        USES_TERMINAL
    )
endforeach()
