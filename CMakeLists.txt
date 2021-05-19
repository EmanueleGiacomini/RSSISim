cmake_minimum_required(VERSION 3.12)
project(RssiSim)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories("./include")

add_subdirectory(src)
add_subdirectory(tests)