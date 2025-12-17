cmake_minimum_required(VERSION 3.15)
project(detect)

add_subdirectory(utils)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PNG_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/libjpeg/build)
include_directories(${PROJECT_SOURCE_DIR}/libjpeg)


set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_executable(${PROJECT_NAME} main.cc)
target_link_libraries(${PROJECT_NAME} PNG::PNG ${PROJECT_SOURCE_DIR}/libjpeg/build/libjpeg.a utils ${MPI_C_LIBRARIES})
