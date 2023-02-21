CMAKE_MINIMUM_REQUIRED(VERSION 3.22)
PROJECT(hmap_generator)

SET(CMAKE_CXX_STANDARD 17)

IF(${TARGET_ARCH} STREQUAL "x86")
    MESSAGE(STATUS "Target architecture: x86")
    # set tengine library directory
    SET(TENGINE_LIB_DIR ${PROJECT_SOURCE_DIR}/3rdparty/tengine/lib/x86)
    # set output directory
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/x86/bin)
    SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/x86/lib)
ELSEIF(${TARGET_ARCH} STREQUAL "arm")
    MESSAGE(STATUS "Target architecture: arm")
    # set tengine & opencv library directory
    SET(TENGINE_LIB_DIR ${PROJECT_SOURCE_DIR}/3rdparty/tengine/lib/aarch64)
    SET(OpenCV_DIR /opt/libs_aarch64/opencv_aarch64/lib/cmake/opencv4)
    # set output directory
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/aarch64/bin)
    SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/aarch64/lib)
    # set compiler
    SET(CMAKE_C_COMPILER /opt/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
    SET(CMAKE_CXX_COMPILER /opt/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++)
ELSE()
    MESSAGE(FATAL_ERROR "Unsupported target architecture: ${TARGET_ARCH}")
ENDIF()

FIND_PACKAGE(OpenCV REQUIRED)

# Build hmap_generator library
FILE(GLOB TENGINE_LIBS ${TENGINE_LIB_DIR}/*.so)

# TODO: temporary test
ADD_LIBRARY(hmap_generator SHARED src/hmap_generator.cpp)
#ADD_LIBRARY(hmap_generator STATIC src/hmap_generator.cpp)

TARGET_INCLUDE_DIRECTORIES(hmap_generator PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/3rdparty/tengine/include
        )
TARGET_LINK_LIBRARIES(hmap_generator PUBLIC
        ${TENGINE_LIBS}
        ${OpenCV_LIBS}
        )

# Build test
ADD_EXECUTABLE(hmap_generator_test main.cpp)
TARGET_LINK_LIBRARIES(hmap_generator_test PUBLIC
        hmap_generator
        ${OpenCV_LIBS}
        )

