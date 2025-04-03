cmake_minimum_required(VERSION 3.20)
project(JobShopScheduler CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Poprawne flagi dla CUDA 12.x
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fmad=false --prec-div=true --prec-sqrt=true")

# Dodaj katalog źródłowy
include_directories(src)

# Pobierz bibliotekę nlohmann/json (jeśli nie masz jej lokalnie)
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2  # Możesz wybrać nowszą wersję
)
FetchContent_MakeAvailable(json)

# Dodaj pliki źródłowe CUDA
file(GLOB_RECURSE CUDA_SOURCES
    src/*.cu
)

# Dodaj pliki źródłowe C++
file(GLOB_RECURSE CPP_SOURCES
    src/*.cpp
    src/*.h
    src/*.hpp
)

# Optionally, you can print the found files for debugging
message(STATUS "Found CUDA sources: ${CUDA_SOURCES}")
message(STATUS "Found C++ sources: ${CPP_SOURCES}")

# Dodaj pliki źródłowe do projektu
add_executable(JobShopScheduler ${CPP_SOURCES} ${CUDA_SOURCES})

# Ustawienia dla CUDA
set_property(TARGET JobShopScheduler PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET JobShopScheduler PROPERTY CUDA_ARCHITECTURES 75)

# Linkowanie bibliotek
target_link_libraries(JobShopScheduler PRIVATE nlohmann_json::nlohmann_json)

file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})