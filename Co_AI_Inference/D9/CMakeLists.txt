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

include(CTest)
if(BUILD_TESTING)
    add_executable(kv_cache_tests tests.cpp)
    set_target_properties(kv_cache_tests PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kv_cache_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME sequence_kv_cache COMMAND kv_cache_tests)
    add_test(NAME kv_cache_scheduling
        COMMAND ${CMAKE_COMMAND} -DKV_CACHE_PROGRAM=$<TARGET_FILE:kv_cache>
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests_cli.cmake
    )
endif()

set(KV_CACHE_RUN_ARGS "" CACHE STRING "Program arguments, separated by semicolons")
add_custom_target(run
    COMMAND $<TARGET_FILE:kv_cache> ${KV_CACHE_RUN_ARGS}
    DEPENDS kv_cache
    USES_TERMINAL
    VERBATIM
)
