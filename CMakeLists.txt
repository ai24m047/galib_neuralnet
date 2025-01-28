cmake_minimum_required(VERSION 3.10)
project(evol_neuralnet)

set(CMAKE_CXX_STANDARD 20)

# Important setup instructions:
# 1. GALib Setup:
#    - Download GALib from: http://lancet.mit.edu/ga/
#    - Build or compile the library separately
#    - Replace the path below with YOUR actual GALib include directory
include_directories("PATH_TO_GALIB_INCLUDE_DIRECTORY")

# 2. Executable configuration
add_executable(evol_neuralnet main.cpp)

# 3. Linking GALib
# Note: Modify the path to point to your specific GALib library file
# Different systems might require different link approaches
target_link_libraries(evol_neuralnet "PATH_TO_GALIB_STATIC_LIBRARY/libgalib.a")
