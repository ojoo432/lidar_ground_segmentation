cmake_minimum_required(VERSION 2.8.3)
project(ground_segmentation)

## Compile as C++11, supported in ROS Kinetic and newer
 add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  sensor_msgs
  utils
  ros_bridge
  communication
  image_labelers
  diff_helpers
  projections
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
  CATKIN_DEPENDS roscpp utils std_msgs ros_bridge
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
   src/ground_segmentation.cpp
 )

 add_executable(ground_segmentation_node src/ground_segmentation.cpp)

 target_link_libraries(ground_segmentation_node
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES}
   ${OpenCV_LIBS}
 )

