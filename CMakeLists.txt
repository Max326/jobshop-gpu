cmake_minimum_required(VERSION 3.20)
project(JobShopScheduler LANGUAGES CXX CUDA)  # Changed order

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

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


# Main executable
add_executable(JobShopScheduler 
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
    ${HEADER_SOURCES})

# CUDA specific settings
set_property(TARGET JobShopScheduler PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET JobShopScheduler PROPERTY CUDA_ARCHITECTURES 75)

# Includes
target_include_directories(JobShopScheduler PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# JSON library
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)
target_link_libraries(JobShopScheduler PRIVATE nlohmann_json::nlohmann_json)

# Data files
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})