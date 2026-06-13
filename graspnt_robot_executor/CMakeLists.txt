cmake_minimum_required(VERSION 3.16)
project(graspnt_robot_executor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/MP /utf-8)
endif()

set(SOURCES
    src/main.cpp
    src/grasp_executor.cpp
    src/protocol.cpp
    src/robot_driver.cpp
    src/safety_checker.cpp
    src/udp_server.cpp
)

add_executable(graspnt_robot_executor ${SOURCES})

target_include_directories(graspnt_robot_executor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ------------------ RealMan / Robotic_Arm SDK ------------------
# 默认目录沿用 door4 的三方库布局。你也可以在 Visual Studio/CMake 配置里覆盖：
# -DROBOTIC_ARM_DIR=D:/path/to/Robotic_Arm
set(ROBOTIC_ARM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/Robotic_Arm" CACHE PATH "RealMan Robotic_Arm SDK root")
set(ROBOTIC_ARM_LIB_DIR "${ROBOTIC_ARM_DIR}/lib")
set(ROBOTIC_ARM_INCLUDE_DIR "${ROBOTIC_ARM_DIR}/include")

find_library(RMAN_API_LIB NAMES api_cpp PATHS "${ROBOTIC_ARM_LIB_DIR}" NO_DEFAULT_PATH)
if(NOT RMAN_API_LIB)
    message(FATAL_ERROR "api_cpp library not found in ${ROBOTIC_ARM_LIB_DIR}. Set ROBOTIC_ARM_DIR to your SDK root.")
endif()
message(STATUS "Found Robotic_Arm library: ${RMAN_API_LIB}")

target_include_directories(graspnt_robot_executor PRIVATE
    ${ROBOTIC_ARM_INCLUDE_DIR}
)
target_link_libraries(graspnt_robot_executor PRIVATE ${RMAN_API_LIB})

find_package(nlohmann_json CONFIG QUIET)
if(nlohmann_json_FOUND)
    target_link_libraries(graspnt_robot_executor PRIVATE nlohmann_json::nlohmann_json)
else()
    message(STATUS "nlohmann_json package not found. Ensure nlohmann/json.hpp is in your include path.")
endif()

if(WIN32)
    target_link_libraries(graspnt_robot_executor PRIVATE ws2_32)
    add_custom_command(TARGET graspnt_robot_executor POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ROBOTIC_ARM_LIB_DIR}/api_cpp.dll"
            $<TARGET_FILE_DIR:graspnt_robot_executor>
        COMMENT "Copying api_cpp.dll to output directory"
    )
endif()
