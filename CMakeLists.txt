cmake_minimum_required(VERSION 3.17)
project(test_tvm)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED core imgproc highgui)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_LIST_DIR}/tvm-0.10.0/include)

#add_library(tvm SHARED IMPORTED)
#set_target_properties(tvm PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_LIST_DIR}/tvm-0.10.0/libtvm.dylib)
add_library(tvm_runtime SHARED IMPORTED)
set_target_properties(tvm_runtime PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/tvm-0.10.0/libtvm_runtime.dylib)


add_executable(test_tvm ultralytics_yolov3/ultralytics_yolov3.cpp)
target_link_libraries(test_tvm ${OpenCV_LIBS})
#target_link_libraries(test_tvm tvm)
target_link_libraries(test_tvm tvm_runtime)

