# CMakeLists.txt
cmake_minimum_required(VERSION 3.8)
project(image_stitcher)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)

add_executable(image_stitcher src/image_stitcher.cpp)
ament_target_dependencies(image_stitcher
  rclcpp
  opencv
  cv_bridge
  sensor_msgs
  message_filters
)

install(TARGETS
  image_stitcher
  DESTINATION lib/${PROJECT_NAME})

ament_package()