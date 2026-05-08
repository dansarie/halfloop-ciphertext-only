# Implementation of attacks on HALFLOOP-24.
#
# Copyright (C) 2022 Marcus Dansarie, Patrick Derbez, Gregor Leander, and
# Lukas Stennes.
# Copyright (C) 2025-2026 Marcus Dansarie, Gregor Leander, Shahram Rasoolzadeh,
# Lukas Stennes, and Cihangir Tezcan.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

cmake_minimum_required (VERSION 3.29)

project(HALFLOOP-COA VERSION 1.0 LANGUAGES C)

option(ENABLE_CUDA "Enable CUDA support" ON)

if (ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
endif()

set(CMAKE_C_FLAGS_RELEASE "")

if (WIN32)
  set(CMAKE_C_FLAGS "/W4 /arch:AVX2")
else ()
  set(CMAKE_C_FLAGS "-march=native -O3 -g -Wall -Wpedantic")
endif()

add_executable(
  halfloop-coa
  getopt.c
  halfloop-bitslice.c
  halfloop-coa.c
  halfloop-common.c)
add_executable(halfloop-generate-data halfloop-generate-data.c halfloop-common.c)

find_library(MATH_LIBRARY m)
if (MATH_LIBRARY)
  target_link_libraries(halfloop-coa PUBLIC ${MATH_LIBRARY})
endif()
if (WIN32)
  target_link_libraries(halfloop-coa PRIVATE bcrypt)
  target_link_libraries(halfloop-generate-data PRIVATE bcrypt)
endif()

if (CUDAToolkit_FOUND)
  enable_language(CUDA)
  message(STATUS "CUDA support enabled.")
  target_sources(halfloop-coa PRIVATE halfloop-cuda.cu)
  set_target_properties(halfloop-coa PROPERTIES CUDA_ARCHITECTURES "native")
  target_compile_definitions(halfloop-coa PRIVATE CUDA_ENABLED=1)
  if (WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xptxas=-v -Wall -Xcompiler /arch:AVX2 -Xcompiler /wd4668 -Xcompiler /wd4820 -Xcompiler /wd5039 -Xcompiler /wd5045 -std=c++20 -Xcompiler \"/std:c++20\"")
  else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -std=c++20 -lineinfo -Xptxas=-v -Wall -march=native -O3")
  endif()
  target_link_libraries(halfloop-coa PUBLIC CUDA::cuda_driver ${CUDA_LIBRARIES})
else()
  message(STATUS "CUDA support disabled.")
endif()
