cmake_minimum_required(VERSION 3.9)
project(sharpy)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(lib)
