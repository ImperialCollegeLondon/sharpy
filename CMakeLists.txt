cmake_minimum_required(VERSION 3.5)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

execute_process(COMMAND "git submodule init && git submodule update"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_subdirectory(lib)
