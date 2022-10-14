PROJECT(spixel)
CMAKE_MINIMUM_REQUIRED(VERSION 3.4)

if(POLICY CMP0011)
  cmake_policy(SET CMP0011 NEW)
endif()
cmake_policy(SET CMP0012 NEW)

### OPENCV

FIND_PACKAGE(OpenCV REQUIRED)

### OPENMP

FIND_PACKAGE(OpenMP)

IF(OPENMP_FOUND)
MESSAGE("OpenMP Enabled")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ENDIF(OPENMP_FOUND)

### PYBIND11 + PYTHON + NUMPY

find_package(pybind11 CONFIG REQUIRED)
MESSAGE( " *** PYTHON_INCLUDE_DIRS : " ${PYTHON_INCLUDE_DIRS} )
MESSAGE( " *** PYTHON_LIBRARIES : " ${PYTHON_LIBRARIES} )

set(NUMPY_INCLUDE_DIR "" CACHE FILEPATH "Path to numpy header if cmake can't find them.")
if (NOT ${NUMPY_INCLUDE_DIR} STREQUAL "")
  message( " *** NUMPY_INCLUDE_DIR : ${NUMPY_INCLUDE_DIR}" )
  if(NOT EXISTS ${NUMPY_INCLUDE_DIR}/numpy/ndarrayobject.h)
    message(SEND_ERROR "Can't find numpy/ndarrayobject.h in ${NUMPY_INCLUDE_DIR}")
    endif()
  include_directories(${NUMPY_INCLUDE_DIR})
endif()

### COMPILER FLAGS

IF(CMAKE_COMPILER_IS_GNUCXX)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -march=native") 
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

### SOURCES + HEADERS

SET(HEADER_FILES src/functions.h src/segengine.h src/stdafx.h src/structures.h src/tsdeque.h src/utils.h src/ndarray_converter.h)
SET(SOURCE_FILES src/stdafx.cpp src/utils.cpp src/functions.cpp src/structures.cpp src/segengine.cpp src/ndarray_converter.cpp)

### INCLUDE DIRS

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS}/opencv4)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

### Linker
LINK_DIRECTORIES(${OpenCV_LIB_DIR})

#**************************************************************************************************
# Make configuration
#**************************************************************************************************
pybind11_add_module(spixel MODULE ${HEADER_FILES} ${SOURCE_FILES} src/python_binding.cpp)
target_compile_features(spixel PRIVATE cxx_std_17)
target_link_libraries(spixel PRIVATE ${OpenCV_LIBS})
