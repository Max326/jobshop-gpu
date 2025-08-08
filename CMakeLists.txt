cmake_minimum_required(VERSION 3.20)

# Set the CUDA architecture globally before the project() call.
# This is the most reliable way to avoid initialization errors.
set(CMAKE_CUDA_ARCHITECTURES 75)

project(JobShopScheduler LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Add CUDA flags to resolve Eigen warnings and improve code generation.
# --expt-relaxed-constexpr fixes the common Eigen/CUDA compatibility issue.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fmad=false --prec-div=true --prec-sqrt=true -lineinfo")

# --- Source File Organization ---
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADER_SOURCES "src/*.h" "src/*.hpp" "src/*.cuh")

# --- Dependency: libcmaes (Modern Approach) ---
# Define the root path for the adjacent libcmaes project.
set(LIBCMAES_ROOT "${CMAKE_SOURCE_DIR}/../libcmaes")

# Find the pre-built libcmaes library file (.so or .a).
find_library(
    LIBCMAES_LIBRARY
    NAMES cmaes libcmaes
    HINTS "${LIBCMAES_ROOT}/src/.libs"
    DOC "Path to the libcmaes library file"
)

# Define the include directories needed for libcmaes.
set(LIBCMAES_INCLUDE_DIRS
    "${LIBCMAES_ROOT}/include"
    "${LIBCMAES_ROOT}/include/libcmaes"
)

# Stop with an error if the library wasn't found.
if(NOT LIBCMAES_LIBRARY)
    message(FATAL_ERROR "libcmaes library not found in ${LIBCMAES_ROOT}/src/.libs. "
                        "Please ensure libcmaes is built.")
endif()
message(STATUS "Found libcmaes library: ${LIBCMAES_LIBRARY}")

# Create an IMPORTED target. This bundles the library location and its include
# directories into a single, modern CMake target.
add_library(cmaes_external UNKNOWN IMPORTED)
set_target_properties(cmaes_external PROPERTIES
    IMPORTED_LOCATION "${LIBCMAES_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBCMAES_INCLUDE_DIRS}"
)

# --- Dependency: Eigen3 ---
# Find Eigen3 using CMake's built-in capabilities.
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# --- Dependency: OpenMP ---
find_package(OpenMP)
if(OPENMP_FOUND)
    message(STATUS "OpenMP found, enabling flags.")
else()
    message(WARNING "OpenMP not found.")
endif()

# --- Main Executable Target ---
add_executable(JobShopScheduler
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
    ${HEADER_SOURCES}
)

# --- Target-Specific Properties ---
# Enable separable compilation for CUDA.
set_property(TARGET JobShopScheduler PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Add include directories for your own source code and Eigen.
target_include_directories(JobShopScheduler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIRS}
)

# --- Dependency: nlohmann_json (via FetchContent) ---
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# --- Linking Stage ---
# Link all libraries to the executable. Linking to 'cmaes_external' will
# also automatically add its required include directories.
target_link_libraries(JobShopScheduler PRIVATE
    nlohmann_json::nlohmann_json
    cmaes_external
    OpenMP::OpenMP_CXX
)

# --- RPATH Configuration for Runtime ---
# Embed the library's path into the executable so the system can find
# libcmaes.so at runtime, avoiding the "cannot open shared object file" error.
set_target_properties(JobShopScheduler PROPERTIES
    INSTALL_RPATH "$<TARGET_FILE_DIR:cmaes_external>"
)

# --- Data Files ---
file(COPY "${CMAKE_SOURCE_DIR}/data" DESTINATION "${CMAKE_BINARY_DIR}")
