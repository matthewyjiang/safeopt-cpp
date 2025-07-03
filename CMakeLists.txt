cmake_minimum_required(VERSION 3.10)
project(SafeOptCpp VERSION 0.16.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# Find required packages
find_package(Eigen3 REQUIRED)

find_package(Matplot++ REQUIRED)

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
    src/visualization.cpp
)

# Link libraries
target_link_libraries(safeopt_cpp
    Eigen3::Eigen
    gp_lib
    Matplot++::matplot
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

add_executable(visualization_example
    examples/visualization_example.cpp
)
target_link_libraries(visualization_example safeopt_cpp)

# Enable testing
enable_testing()

# Add test executable
add_executable(test_safeopt
    tests/test_all.cpp
)
target_link_libraries(test_safeopt safeopt_cpp)

# Add 3D optimization test
add_executable(test_3d_optimization
    tests/test_3d_optimization.cpp
)
target_link_libraries(test_3d_optimization safeopt_cpp Matplot++::matplot)

# Add visualization test
add_executable(test_visualization
    tests/test_visualization.cpp
)
target_link_libraries(test_visualization safeopt_cpp)

# Add real function optimization test
add_executable(test_real_function_optimization
    tests/test_real_function_optimization.cpp
)
target_link_libraries(test_real_function_optimization safeopt_cpp)

# Add GP demo executable (commented out - file doesn't exist yet)
# add_executable(test_gp_demo
#     test_gp_demo.cpp
# )
# target_link_libraries(test_gp_demo gp_lib)

# Add tests
add_test(NAME SafeOptTests COMMAND test_safeopt)
add_test(NAME SafeOpt3DTest COMMAND test_3d_optimization)
add_test(NAME VisualizationTest COMMAND test_visualization)
add_test(NAME RealFunctionOptimizationTest COMMAND test_real_function_optimization)

# Install targets
install(TARGETS safeopt_cpp
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)
