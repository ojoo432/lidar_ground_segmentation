cmake_minimum_required(VERSION 2.8.3)
project(image_labelers)

## Compile as C++11, supported in ROS Kinetic and newer
 add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  projections
  utils
  diff_helpers
  sensor_msgs
  std_msgs
  roslib
  tf
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3	 REQUIRED)
find_package(PCL REQUIRED COMPONENTS )
find_package(OpenCV REQUIRED)


catkin_package(
  INCLUDE_DIRS include
  LIBRARIES utils diff_helpers
#  CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)


include_directories(
 include
  src
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare a C++ library
 add_library(${PROJECT_NAME}
   src/abstract_image_labeler.cpp
 )

add_dependencies(image_labelers ${catkin_EXPORTED_TARGETS})


