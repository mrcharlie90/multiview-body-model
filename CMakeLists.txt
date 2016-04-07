cmake_minimum_required(VERSION 3.5)
project(helloworld)

find_package(PCL 1.7)

add_definitions(${PCL_DEFINITIONS})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

find_package( OpenCV REQUIRED )

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_executable(main main.cpp )
target_link_libraries(main ${OpenCV_LIBS} ${PCL_LIBRARIES})

MESSAGE(${PCL_LIBRARIES})