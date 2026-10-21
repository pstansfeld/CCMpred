cmake_minimum_required(VERSION 3.18)
include(CheckLanguage)

project(libconjugrad LANGUAGES C)
set(LIBCONJUGRAD_MAJOR_VERSION 0)
set(LIBCONJUGRAD_MINOR_VERSION 0)
set(LIBCONJUGRAD_PATCH_VERSION 4)
set(LIBCONJUGRAD_VERSION ${LIBCONJUGRAD_MAJOR_VERSION}.${LIBCONJUGRAD_MINOR_VERSION}.${LIBCONJUGRAD_PATCH_VERSION})

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_EXTENSIONS ON)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

include_directories("${PROJECT_SOURCE_DIR}/include")

set(SOURCES src/conjugrad.c src/debug.c)

if(NOT DEFINED WITH_SIMD)
	set(WITH_SIMD ansi)
endif(NOT DEFINED WITH_SIMD)

if(NOT DEFINED WITH_CUDA)
	set(WITH_CUDA ON)
endif(NOT DEFINED WITH_CUDA)

if(WITH_SIMD STREQUAL ansi)
	add_definitions(-DCONJUGRAD_SIMD=0)
	list(APPEND SOURCES src/arithmetic_ansi.c)
elseif(WITH_SIMD STREQUAL sse)
	add_definitions(-DCONJUGRAD_SIMD=1)
	list(APPEND SOURCES src/arithmetic_simd.c)
elseif(WITH_SIMD STREQUAL avx)
	add_definitions(-DCONJUGRAD_SIMD=2)
	list(APPEND SOURCES src/arithmetic_simd.c)
endif(WITH_SIMD STREQUAL ansi)

set(BUILD_SHARED_LIBS OFF)

if(WITH_CUDA)
	check_language(CUDA)
	if(NOT CMAKE_CUDA_COMPILER)
		set(WITH_CUDA OFF)
	endif()
endif(WITH_CUDA)
if(WITH_CUDA)
	enable_language(CUDA)

	#list(APPEND SOURCES src/conjugrad_cuda.c src/help_functions.cu src/conjugrad_kernels.cu)
	list(APPEND SOURCES src/conjugrad_cuda.c src/conjugrad_kernels.cu)

	include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif(WITH_CUDA)

add_library(conjugrad ${SOURCES})
find_package(OpenMP COMPONENTS C)
if(OpenMP_C_FOUND)
	target_link_libraries(conjugrad OpenMP::OpenMP_C)
endif()
if(UNIX)
	target_link_libraries(conjugrad m)
endif(UNIX)

install(TARGETS conjugrad DESTINATION lib)

add_executable(rosenbrock EXCLUDE_FROM_ALL samples/rosenbrock.c)
target_link_libraries(rosenbrock conjugrad)

add_executable(polynomial EXCLUDE_FROM_ALL samples/polynomial.c)
target_link_libraries(polynomial conjugrad)

add_executable(sphere EXCLUDE_FROM_ALL samples/sphere.c)
target_link_libraries(sphere conjugrad)
