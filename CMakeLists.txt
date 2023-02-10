CMAKE_MINIMUM_REQUIRED(VERSION 3.22)
PROJECT(hmap_generator)

SET(CMAKE_CXX_STANDARD 17)
FIND_PACKAGE(OpenCV REQUIRED)

# Build hmap_generator library
FILE(GLOB TENGINE_LIBS ${PROJECT_SOURCE_DIR}/3rdparty/tengine/lib/x86/*.so)
MESSAGE(STATUS "Tengine libs: ${TENGINE_LIBS}")
ADD_LIBRARY(hmap_generator SHARED src/hmap_generator.cpp)
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

