cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(benchmark)
set(CMAKE_CXX_STANDARD 14)
list(APPEND CMAKE_PREFIX_PATH "/home/baykeremm/thesis/libtorch")
list(APPEND CMAKE_PREFIX_PATH "/home/baykeremm/thesis/libtorch/share/cmake")

find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(${PROJECT_NAME} main.cpp handler3d.cpp handler3d.hpp yolo.cpp yolo.hpp rektnet.hpp rektnet.cpp utils.hpp)

target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
