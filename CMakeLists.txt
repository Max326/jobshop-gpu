cmake_minimum_required(VERSION 3.20)
project(JobShopScheduler LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Better CUDA flags for modern architectures
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fmad=false --prec-div=true --prec-sqrt=true -lineinfo")

# Source organization
file(GLOB_RECURSE CUDA_SOURCES
    src/*.cu
)

file(GLOB_RECURSE CPP_SOURCES
    src/*.cpp
)

file(GLOB_RECURSE HEADER_SOURCES
    src/*.h
    src/*.hpp
    src/*.cuh
)

# General include for Eigen3 (remains unchanged)
include_directories("/usr/include/eigen3")

# ------------------- LIBCMAES Configuration Start -------------------
# Define the root directory of your pre-built libcmaes
# ${CMAKE_SOURCE_DIR} is Projects/jobshop-gpu/
set(LIBCMAES_EXTERNAL_ROOT "${CMAKE_SOURCE_DIR}/../libcmaes")

# Find the libcmaes static library
find_library(LIBCMAES_LIBRARY_FILE
             NAMES cmaes libcmaes # CMake will search for libcmaes.a or cmaes.a
             HINTS "${LIBCMAES_EXTERNAL_ROOT}/src/.libs"
             NO_DEFAULT_PATH) # Only search in the specified HINTS path

if(NOT LIBCMAES_LIBRARY_FILE)
    message(FATAL_ERROR "libcmaes library (.a) not found in ${LIBCMAES_EXTERNAL_ROOT}/src/.libs. "
                        "Please ensure libcmaes is built there (run make in Projects/libcmaes) and the path is correct.")
endif()
message(STATUS "Found pre-built libcmaes library: ${LIBCMAES_LIBRARY_FILE}")

# Define the include directories for libcmaes
# These will be associated with the imported target.
set(LIBCMAES_INCLUDE_DIRS
    "${LIBCMAES_EXTERNAL_ROOT}/include"
    "${LIBCMAES_EXTERNAL_ROOT}/include/libcmaes" # Keep both as per your original setup
)

# Verify include paths exist (optional, good for debugging)
foreach(include_dir ${LIBCMAES_INCLUDE_DIRS})
    if(NOT IS_DIRECTORY ${include_dir})
        message(WARNING "libcmaes include directory not found: ${include_dir}")
    else()
        message(STATUS "Found libcmaes include directory: ${include_dir}")
    endif()
endforeach()

# Create an IMPORTED target for libcmaes
add_library(cmaes_external STATIC IMPORTED) # You can make it GLOBAL if needed across subdirectories
set_target_properties(cmaes_external PROPERTIES
    IMPORTED_LOCATION "${LIBCMAES_LIBRARY_FILE}"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBCMAES_INCLUDE_DIRS}" # Semicolon-separated list
)
# -------------------- LIBCMAES Configuration End --------------------

# CXX Flags for JobShopScheduler project
set(CMAKE_CXX_FLAGS "-O3") # Your base CXX flags for this project
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DNDEBUG") # Note: NDEBUG usually for release
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native") # Add march=native

# OpenMP configuration (remains unchanged)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Main executable
add_executable(JobShopScheduler
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
    ${HEADER_SOURCES})

# CUDA specific settings (remains unchanged)
set_property(TARGET JobShopScheduler PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET JobShopScheduler PROPERTY CUDA_ARCHITECTURES 75)

# Includes for JobShopScheduler sources (remains unchanged)
target_include_directories(JobShopScheduler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# JSON library (remains unchanged)
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# Modified linking: link against the imported cmaes_external target
target_link_libraries(JobShopScheduler PRIVATE
    nlohmann_json::nlohmann_json
    cmaes_external # Link against the imported libcmaes target
)

# Data files (remains unchanged)
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

