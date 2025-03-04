import rosbag
import numpy as np
import matplotlib.pyplot as plt

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

topics = ["/mpc_solution", "/force_input"]

bag_path = '/home/fruscelli/Documents/concert_obstacles/record_concert/obstacles_data_2024-10-23_11-19-03/bag_2024-10-23_11-19-03.bag'
bag = rosbag.Bag(bag_path)

# print(bag)
# exit()
times = []
forces_x = []
forces_y = []
vels = []
qs = []

lidar_obs_list = []
lidar_robot_list = []
# lidar_robot_sphere_x = []
# lidar_robot_sphere_y = []


for topic, msg, t in bag.read_messages(topics='/mpc_solution'):
    times.append(t.to_sec())
    vels.append(msg.v)
    qs.append(msg.q)

times = [elem - times[0] for elem in times]
for topic, msg, t in bag.read_messages(topics='/force_input'):
    forces_x.append(msg.wrench.force.x)
    forces_y.append(msg.wrench.force.y)
    # forces.append(msg.force.x)
    # forces.append(msg.force.y)

t_matrix = np.array(times)
v_matrix = np.array(vels)
q_matrix = np.array(qs)
forces_x_matrix = np.array(forces_x)
forces_y_matrix = np.array(forces_y)

for topic, msg, t in bag.read_messages(topics=['/costmap_node/costmap/costmap/obstacles']):
    lidar_obs_t = []
    if len(msg.markers) > 1:

        for marker in msg.markers:
            lidar_obs_t.append(marker.pose.position)

    lidar_obs_list.append(lidar_obs_t)

for topic, msg, t in bag.read_messages(topics=['/velodyne_map_publisher/robot_markers']):
    lidar_robot_t = []
    for marker in msg.markers:
        lidar_robot_t.append(marker.pose.position)

    lidar_robot_list.append(lidar_robot_t)
    # lidar_robot_sphere_x.append(msg.markers[0].pose.position.x)
    # lidar_robot_sphere_y.append(msg.markers[0].pose.position.y)


min_distances_xy = []
for robot_at_t, spheres_marker_at_t in zip(qs, lidar_obs_list):

    distances_xy_t = []
    # Compute pairwise distances between points in the two frames (only x and y)
    for p_obs in spheres_marker_at_t:
        distances_xy_t.append([np.linalg.norm(np.array([robot_at_t[0] + 0.3, robot_at_t[1]]) - np.array([p_obs.x, p_obs.y]))])

    # Find the minimum distance for the current frame
    if distances_xy_t:
        min_distances_xy.append(min(distances_xy_t))
    else:
        min_distances_xy.append([])



fig, axes = plt.subplots(3, 1, figsize=(10, 10))

# fig.suptitle('test title', fontsize=20)
# plt.xlabel('xlabel', fontsize=18)
# plt.ylabel('ylabel', fontsize=16)
start_obstacle_zone = 6.2
end_obstacle_zone = 10.5
axes[0].plot(times, forces_x_matrix[:-1], label="$F_{ee}^x$", linewidth=2)
axes[0].plot(times, forces_y_matrix[:-1], label="$F_{ee}^y$", linewidth=2)
axes[0].axvspan(start_obstacle_zone, end_obstacle_zone, alpha=0.1, color='red')
axes[0].set_ylabel('F [Nm]')
axes[0].set_xlim([0, 20])

axes[1].plot(times, v_matrix[:, 0], label="$v_{base}^x$", linewidth=2)
axes[1].plot(times, v_matrix[:, 1], label="$v_{base}^y$", linewidth=2)
axes[1].axvspan(start_obstacle_zone, end_obstacle_zone, alpha=0.1, color='red')
axes[1].set_ylabel('v [m/s]')
axes[1].set_xlim([0, 20])

axes[2].plot(times[1::2], min_distances_xy[1::2], label="closest robot-obstacle distance", linewidth=2)
line_min_distance = 0.9 * np.ones([len(times)])
axes[2].plot(times, line_min_distance, label="distance threshold", linestyle = "dashed", color="tab:red", linewidth=2)
axes[2].axvspan(start_obstacle_zone, end_obstacle_zone, alpha=0.1, color='red')
axes[2].set_ylabel('d [m]')
axes[2].set_xlabel('T [s]')
axes[2].set_xlim([0, 20])

axes[0].legend()
axes[0].grid()

axes[1].legend()
axes[1].grid()

axes[2].legend()
axes[2].grid()


plt.tight_layout()
# plt.show()


plt.savefig('/home/fruscelli/Documents/obs_distances.png', bbox_inches='tight')