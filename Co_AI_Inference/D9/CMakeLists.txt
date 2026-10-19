cmake_minimum_required(VERSION 3.16)
project(d9_kv_cache LANGUAGES CXX)

add_executable(kv_cache main.cpp)
set_target_properties(kv_cache PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kv_cache PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(KV_CACHE_RUN_ARGS "" CACHE STRING "Program arguments, separated by semicolons")
add_custom_target(run
    COMMAND $<TARGET_FILE:kv_cache> ${KV_CACHE_RUN_ARGS}
    DEPENDS kv_cache
    USES_TERMINAL
    VERBATIM
)
