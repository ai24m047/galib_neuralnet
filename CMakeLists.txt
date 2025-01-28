cmake_minimum_required(VERSION 3.10)
project(evol_neuralnet)

set(CMAKE_CXX_STANDARD 20)

# Important setup instructions:
# 1. GALib Setup:
#    - Download GALib from: http://lancet.mit.edu/ga/
#    - Build or compile the library separately
#    - Replace the path below with YOUR actual GALib include directory
include_directories("/usr/local/include/ga")

# 2. Executable configuration
add_executable(evol_neuralnet main.cpp)

# 3. Linking GALib
# Note: Modify the path to point to your specific GALib library file
# Different systems might require different link approaches
target_link_libraries(evol_neuralnet "/usr/local/lib/libgalib.a")

# Check if python3 exists, if not, try python
find_program(PYTHON3_EXECUTABLE python3)
if(NOT PYTHON3_EXECUTABLE)
    find_program(PYTHON_EXECUTABLE python)
    if(NOT PYTHON_EXECUTABLE)
        message(FATAL_ERROR "Neither python3 nor python exists")
    else()
        set(PYTHON3_EXECUTABLE ${PYTHON_EXECUTABLE})
    endif()
endif()

# Test target
add_custom_target(test
        COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR} ./evol_neuralnet
        DEPENDS evol_neuralnet
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
