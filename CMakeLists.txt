PROJECT(spixel)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(OpenMP) 

IF(OPENMP_FOUND)
    MESSAGE("OpenMP Enabled")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ENDIF(OPENMP_FOUND)

SET(HEADER_FILES spixel/functions.h spixel/segengine.h spixel/stdafx.h spixel/structures.h spixel/targetver.h spixel/tsdeque.h spixel/utils.h)
SET(SOURCE_FILES spixel/stdafx.cpp spixel/utils.cpp spixel/functions.cpp spixel/structures.cpp spixel/segengine.cpp spixel/spixel.cpp)


INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
ADD_EXECUTABLE(spixel ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(spixel ${OpenCV_LIBS})
