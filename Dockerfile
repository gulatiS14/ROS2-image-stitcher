# Use ROS2 Humble as the base image
FROM ros:humble

# Install required dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libopencv-dev \
    ros-humble-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# Create the /app directory
RUN mkdir /app

# Set up the workspace
WORKDIR /ros2_ws/src/image_stitcher

# Copy the package files
COPY CMakeLists.txt package.xml ./
COPY src ./src

# Build the package
WORKDIR /ros2_ws
RUN . /opt/ros/humble/setup.sh && \
    colcon build --packages-select image_stitcher

# Set up the entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "run", "image_stitcher", "image_stitcher"]