cmake_minimum_required(VERSION 3.10)
project(SafeOptCpp VERSION 0.16.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# Find required packages
find_package(Eigen3 REQUIRED)

# Fetch GP library
include(FetchContent)
FetchContent_Declare(
    gaussian_process
    GIT_REPOSITORY https://github.com/matthewyjiang/gaussian-process.git
    GIT_TAG main
)
FetchContent_MakeAvailable(gaussian_process)

# Add include directories
include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Add library
add_library(safeopt_cpp
    src/utilities.cpp
    src/gaussian_process_optimization.cpp
    src/safe_opt.cpp
    src/safe_opt_swarm.cpp
    src/swarm_optimization.cpp
)

# Link libraries
target_link_libraries(safeopt_cpp
    Eigen3::Eigen
    gp_lib
)

# Add executable for examples
add_executable(safeopt_example
    examples/basic_example.cpp
)
target_link_libraries(safeopt_example safeopt_cpp)

add_executable(safeopt_demo
    examples/comprehensive_demo.cpp
)
target_link_libraries(safeopt_demo safeopt_cpp)

# Enable testing
enable_testing()

# Add test executable
add_executable(test_safeopt
    tests/test_all.cpp
)
target_link_libraries(test_safeopt safeopt_cpp)

# Add GP demo executable
add_executable(test_gp_demo
    test_gp_demo.cpp
)
target_link_libraries(test_gp_demo gp_lib)

# Add test
add_test(NAME SafeOptTests COMMAND test_safeopt)

# Install targets
install(TARGETS safeopt_cpp
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)