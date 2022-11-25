cmake_minimum_required (VERSION 3.0)
project (TsXFilterApplication)

LIST(APPEND CMAKE_CXX_FLAGS "-march=native -ffast-math -fopenmp -Wall -pedantic -Wextra -std=c++17 -Wno-int-in-bool-context -Wno-class-memaccess -Wno-unused-parameter -Wno-deprecated-copy")
LIST(APPEND CMAKE_C_FLAGS "-march=native -ffast-math -fopenmp -Wall -pedantic -Wextra -Wno-int-in-bool-context -Wno-unused-parameter")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -O3 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS} -O3 -g -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -O0 -g")

   
find_package(TIFF REQUIRED)

add_executable(applicationCPU
    source/main.cpp
    source/FloatImage.cpp
    source/FloatImage.h
    source/RunFilter.cpp
    source/RunFilter.h
    source/S2S_net_TsX_network.c
    source/S2S_net_TsX_network.h
    source/S2S_net_TsX_parameters.c
)


target_include_directories(applicationCPU PUBLIC
    ${TIFF_INCLUDE_DIRS}
)

target_link_libraries(applicationCPU
    ${TIFF_LIBRARIES}
)

