# CMake version
cmake_minimum_required (VERSION 3.9)
MESSAGE(STATUS "CMAKE_ROOT: " ${CMAKE_ROOT})

# Project name
project(OPT_cudaGraph VERSION 1.0.0 LANGUAGES CXX CUDA)

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)

option(OPT_BUILD_CUDA "Enables builds of cuda code" OFF)

if(OPT_BUILD_CUDA)
  message(STATUS "Configuring CUDA ...")
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "\nNo CUDA compiler found")
  endif()
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11")
    message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(FATAL_ERROR "\nTaskflow requires CUDA at least v11")
  endif()
  
  # configure cuBLAS
  message(STATUS "Configuring CUDA BLAS ...")
  list(APPEND CMAKE_MODULE_PATH ${TF_CMAKE_DIR}/cuda/modules/)
  find_package(cuBLAS)
  message(STATUS "CUBLAS_FOUND: ${CUBLAS_FOUND}")
  message(STATUS "CUBLAS_INCLUDE_DIRS: ${CUBLAS_INCLUDE_DIRS}")
  message(STATUS "CUBLAS_LIBRARIES: ${CUBLAS_LIBRARIES}")
  include_directories(${CUBLAS_INCLUDE_DIRS})
endif()

## defult release build
#set(SNIG_DEFAULT_BUILD_TYPE "Release")
#if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  #message(STATUS "Setting build type to '${SNIG_DEFAULT_BUILD_TYPE}'")
  #set(
    #CMAKE_BUILD_TYPE "${SNIG_DEFAULT_BUILD_TYPE}" 
    #CACHE
    #STRING "Choose the type of build." 
    #FORCE
  #)
  ## Set the possible values of build type for cmake-gui
  #set_property(
    #CACHE 
    #CMAKE_BUILD_TYPE 
    #PROPERTY STRINGS
    #"Debug" "Release"
  #)
#endif()

#not test yet
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.0")
	  message(FATAL_ERROR "\n SNIG requires g++ at least v5.0")
  endif()
### clang++
#elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  #if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0")
		#message(FATAL_ERROR "\n SNIG requires clang++ at least v6.0")
  #endif() 
endif()

# Args
option(OPT_BUILD_TESTS "Enables build of tests" ON)

# installation path
#set(OPT_UTEST_DIR ${PROJECT_SOURCE_DIR}/unittests)
set(OPT_3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd-party)
set(OPT_BENCHMARKS_DIR ${PROJECT_SOURCE_DIR}/benchmarks)

# flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}" "-Wall -Wextra -Wfatal-errors")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}" "-O2 -march=native")

set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--extended-lambda -w")
#set(CMAKE_CUDA_FLAGS_DEBUG ${CMAKE_CUDA_FLAGS_DEBUG} "-lineinfo")
#set(CMAKE_CUDA_FLAGS_RELEASE ${CMAKE_CUDA_FLAGS_RELEASE} "-O2 -w")

# CXX target properties
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES) 
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)

find_package(CUDA QUIET REQUIRED)

# message
message(STATUS "CMAKE_HOST_SYSTEM: ${CMAKE_HOST_SYSTEM}")
message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
message(STATUS "CMAKE_CXX_COMPILER_VERSION: " ${CMAKE_CXX_COMPILER_VERSION})
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "CMAKE_EXE_LINKER_FLAGS: " ${CMAKE_EXE_LINKER_FLAGS})
message(STATUS "CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX})
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_PREFIX_PATH: " ${CMAKE_PREFIX_PATH})
message(STATUS "PROJECT_NAME: " ${PROJECT_NAME})

#include 3rd-party
include_directories(${OPT_3RD_PARTY_DIR})

#include benchmarks
include_directories(${OPT_BENCHMARKS_DIR}/simple_graph)
include_directories(${OPT_BENCHMARKS_DIR}/IEEE_Graph_Challenge)

#cuda
#include_directories(/usr/local/cuda/include/)
#link_directories(/usr/local/cuda/lib64/)


#--------interface------
add_library(${PROJECT_NAME} INTERFACE)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/> 
)
#-----------------------

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# add executables
message(STATUS "building executables ...")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(simple_graph ${PROJECT_SOURCE_DIR}/main/simple_graph.cu)
add_executable(snig ${PROJECT_SOURCE_DIR}/main/snig.cu)
target_link_libraries(simple_graph ${PROJECT_NAME} stdc++fs)
target_link_libraries(snig ${PROJECT_NAME} stdc++fs)


if(NOT CUDA_ARCHITECTURES)
  set_property(TARGET snig PROPERTY CUDA_ARCHITECTURES OFF)
  set_property(TARGET simple_graph PROPERTY CUDA_ARCHITECTURES OFF)
endif()
