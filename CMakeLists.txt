cmake_minimum_required(VERSION 3.12)
project(RssiSim)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories("./include")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE")

message(${PROJECT_SOURCE_DIR})

add_subdirectory(src)
add_subdirectory(tests)