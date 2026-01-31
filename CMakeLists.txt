cmake_minimum_required(VERSION 3.20)
project(GNCProject)

# --ARIA
# set when configuring -> -DARIA_ROOT="C:\Program Files\MobileRobots\Aria"
set(ARIA_ROOT "" CACHE PATH "Path to ARIA install root")

find_path(ARIA_INCLUDE_DIR
    NAMES Aria.h
    PATHS "${ARIA_ROOT}/include"
)

find_library(ARIA_LIBRARY
    NAMES AriaVC15
    PATHS "${ARIA_ROOT}/lib"
)

if(NOT ARIA_INCLUDE_DIR OR NOT ARIA_LIBRARY)
    message(FATAL_ERROR
    "Could not find Aria")
endif()

add_executable(follow_wp_kfposvel
    src/main.cpp
)

target_include_directories(follow_wp_kfposvel PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen-5.0.1"
    "${ARIA_INCLUDE_DIR}"
)

target_link_libraries(follow_wp_kfposvel PRIVATE
    "${ARIA_LIBRARY}"
)

set(Aria_dLL "${ARIA_ROOT}/BIN/Aria.dll")
if(EXISTS "${ARIA_DLL}")
    add_custom_command(TARGET follow_wp_kfposvel POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ARIA_DLL}"
        $<TARGET_FILE_DIR:follow_wp_kfposvel>
)
endif()
