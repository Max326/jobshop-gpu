cmake_minimum_required(VERSION 3.20)
project(JobShopScheduler LANGUAGES CXX CUDA)

# Ustawienia standardów
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Flagi dla CUDA (GTX 1050 Ti = sm_61)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_61 --expt-relaxed-constexpr")

# Flagi debugowania
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo")
endif()

# Biblioteka JSON
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# Źródła (używaj jawnej listy zamiast GLOB dla lepszej kontroli)
set(SOURCES
    src/main.cpp
    src/NeuralNetwork.cu
    src/JobShopHeuristic.cu
)

# Pliki nagłówkowe
set(HEADERS
    src/NeuralNetwork.h
    src/JobShopHeuristic.cuh
    src/JobShopData.h
    src/FileManager.h
)

# Plik wykonywalny
add_executable(JobShopScheduler ${SOURCES} ${HEADERS})

# Linkowanie
target_link_libraries(JobShopScheduler PRIVATE nlohmann_json::nlohmann_json)

# Kopiowanie danych
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})