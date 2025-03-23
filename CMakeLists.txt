cmake_minimum_required(VERSION 3.10)
project(JobShopScheduler CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Dodaj katalog źródłowy
include_directories(src)

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
set_property(TARGET JobShopScheduler PROPERTY CUDA_ARCHITECTURES 75)  # Ustaw odpowiednią architekturę dla Twojego GPU

# Linkowanie bibliotek
target_link_libraries(JobShopScheduler)