cmake_minimum_required(VERSION 3.13)
project(forward_kinematics)
set(CMAKE_CXX_FLAGS "-Wall -fopenmp -O2")
add_executable(out main.cpp)
# Link OpenMP
target_link_libraries(out ${OpenMP_CXX_LIBRARIES})
# Link MPI
find_package(MPI REQUIRED)
target_link_libraries(out MPI::MPI_CXX)
# Link Google benchmark
find_package(benchmark REQUIRED)
target_link_libraries(out benchmark::benchmark)
