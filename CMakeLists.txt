cmake_minimum_required(VERSION 3.30)
project(evol_neuralnet)

set(CMAKE_CXX_STANDARD 20)

include_directories("C:/Users/elisa/CLionProjects/galib/")

add_executable(evol_neuralnet main.cpp)

target_link_libraries(evol_neuralnet "C:/Users/elisa/CLionProjects/galib/cmake-build-debug/libgalib.a")
target_link_libraries(evol_neuralnet "C:/Users/elisa/CLionProjects/galib/cmake-build-release/libgalib.a")
