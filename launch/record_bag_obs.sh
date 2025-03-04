#!/bin/bash

# Default bag name if no name is provided
# Get the current date and time in the format: YYYY-MM-DD_HH-MM-SS
CURRENT_DATE_TIME=$(date +"%Y-%m-%d_%H-%M-%S")

DIR_NAME="obstacles_data_$CURRENT_DATE_TIME"
DEFAULT_BAG_NAME="bag_$CURRENT_DATE_TIME"

mkdir "$DIR_NAME"

# Check if a bag name was provided as an argument
if [ $# -eq 0 ]; then
    BAG_NAME="$DIR_NAME/$DEFAULT_BAG_NAME"
else
    BAG_NAME="$DIR_NAME/$1"
fi


# Record the rosbag with the given name and topics
echo "Recording ROS bag with name: $BAG_NAME"
echo "Recording topics: $TOPICS"

TOPICS="/VLP16_lidar_back/pointcloud_to_scan /VLP16_lidar_back/velodyne_points /VLP16_lidar_front/pointcloud_to_scan /VLP16_lidar_front/velodyne_points /costmap_node/costmap/costmap /costmap_node/costmap/costmap/obstacles /force_marker /joint_states /joy /mpc_solution /sonar_map/obstacles /sonar_map/fl_lat /sonar_map/fl_sag /sonar_map/fr_lat /sonar_map/fr_sag /sonar_map/rl_lat /sonar_map/rl_sag /sonar_map/rr_lat /sonar_map/rr_sag /sonar_map_publisher/robot_markers /velodyne_map_publisher/robot_markers /tf /tf_static /xbotcore/robot_description /xbotcore/command /xbotcore/joint_states"

rosbag record -O "$BAG_NAME" $TOPICS
rosparam dump "$BAG_NAME/robot_description.yaml" /xbotcore/robot_description







