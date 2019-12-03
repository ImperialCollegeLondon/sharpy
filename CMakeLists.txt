cmake_minimum_required(VERSION 3.9)
project(sharpy)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

execute_process(
    COMMAND git submodule init
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
execute_process(
    COMMAND git submodule update
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_subdirectory(lib)
