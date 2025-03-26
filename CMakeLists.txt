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
set(CUDA_SOURCES
    src/NeuralNetwork.cu
)

# Dodaj pliki źródłowe C++
set(CPP_SOURCES
    src/JobShopData.h
    src/JobShopHeuristic.cpp
    src/JobShopHeuristic.h
    src/main.cpp
)

# Dodaj pliki źródłowe do projektu
add_executable(JobShopScheduler ${CPP_SOURCES} ${CUDA_SOURCES})

# Ustawienia dla CUDA
set_property(TARGET JobShopScheduler PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET JobShopScheduler PROPERTY CUDA_ARCHITECTURES 75)

# Linkowanie bibliotek
target_link_libraries(JobShopScheduler PRIVATE nlohmann_json::nlohmann_json)

file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})