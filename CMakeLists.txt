cmake_minimum_required(VERSION 3.5)
project(multiview-body-model)

find_package( OpenCV REQUIRED )

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_executable(multiviewbodymodel main.cpp MultiviewBodyModel.cpp MultiviewBodyModel.h)
target_link_libraries(multiviewbodymodel ${OpenCV_LIBS})
