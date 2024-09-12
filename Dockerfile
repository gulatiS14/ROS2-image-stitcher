# Use ROS2 Humble as the base image
FROM ros:humble

# Install required dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libopencv-dev \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    ros-humble-image-transport \
    ros-humble-message-filters \
    && rm -rf /var/lib/apt/lists/*

# Create a ROS workspace
WORKDIR /ros2_ws/src

# Copy the package files
COPY . /ros2_ws/src/image_stitcher

# Build the package
WORKDIR /ros2_ws
RUN . /opt/ros/humble/setup.sh && \
    colcon build --packages-select image_stitcher

# Set up the entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "run", "image_stitcher", "image_stitcher"]