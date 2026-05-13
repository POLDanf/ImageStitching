cmake_minimum_required(VERSION 3.10)
project(SLAM_Test)

# 1. Compiler and Standards
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)

# 2. DATA DIRECTORY SETUP (Docker Optimized)
# CMAKE_CURRENT_SOURCE_DIR points to the directory containing this CMakeLists.txt.
# This ensures that even if you build in /build, it correctly finds /data in the parent.
set(DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data" CACHE PATH "Path to SLAM datasets")

# 3. Dependencies
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# 4. Create Executable
add_executable(cv_test stitching.cpp)

# 5. Configuration & Includes
# This passes the absolute path to your C++ code as a macro (DATA_PATH)
target_compile_definitions(cv_test PRIVATE DATA_PATH="${DATA_DIR}")
target_include_directories(cv_test PRIVATE ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

# 6. Linking
target_link_libraries(cv_test PRIVATE ${OpenCV_LIBS})