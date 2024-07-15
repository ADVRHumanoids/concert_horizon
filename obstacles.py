#!/usr/bin/python3

from visualization_msgs.msg import Marker
import rospy
import numpy as np
import horizon.utils as utils
from horizon_navigation.pyObstacleGenerator import ObstacleGenerator
from horizon_navigation.pyObstacle import CasadiObstacle, SphereObstacle
import time
from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init

class ObstacleGeneratorWrapper:

    def __init__(self, prb, model, kin_dyn):

        self.prb = prb
        self.model = model
        self.kin_dyn = kin_dyn

        self.obs_gen = None

        self.max_obs_num = 50
        self.radius_sphere_robot_x = 1.
        self.radius_sphere_robot_y = 0.8
        self.radius_sphere_robot_z = 0.01

        self.f_obs_grid = 0  # function of the inputs
        self.obstacle_radius = 0.15
        self.weight_cost_obs = 0.002  # 0.001 # 0.0025
        self.angle_threshold = 0.2
        self.min_blind_angle = -np.pi/6
        self.max_blind_angle = - self.min_blind_angle

        self.occupancy_map_width = 6.0
        self.occupancy_map_height = 6.0
        self.occupancy_map_resolution = 0.01
        self.occupancy_map_topic_names = ["/navigation_map", "/sonar_map"]

        self.obs_origin_par_list = [] # list of origin parameters for each obstacle
        self.obs_weight_par_list = [] # list of weight parameters for each obstacle

        # self.time_obstacles_list = list()

        self.__init_obstacle_generator()
        self.__init_markers()
        self.__init_ros_publisher()

        self.base_fk = kin_dyn.fk('base_link')

    def __init_obstacle_generator(self):

        self.obs_gen = ObstacleGenerator(self.occupancy_map_width,
                                         self.occupancy_map_height,
                                         self.occupancy_map_resolution,
                                         self.occupancy_map_topic_names)


        self.obs_gen.setMaxObstacleNum(self.max_obs_num) # maximum number of obstacle allowed
        self.obs_gen.setObstacleRadius(self.obstacle_radius) # radius of obstacle
        self.obs_gen.setAngleThreshold(self.angle_threshold) # angle resolution to remove unnecessary obstacles
        self.obs_gen.setBlindAngle(self.min_blind_angle, self.max_blind_angle) # blindsight of the robot, does not consider obstacles

        # obs_fun = CasadiObstacle().simpleFormulation()

        grid_origin = self.kin_dyn.fk('base_link')(q=self.model.q)['ee_pos'][:2]
        radius_robot = np.array([self.radius_sphere_robot_x, self.radius_sphere_robot_y])


        for obs_num in range(self.max_obs_num):

            # add to cost function all the casadi obstacles, parametrized with ORIGIN and WEIGHT
            obs_origin_par = self.prb.createParameter(f'obs_origin_{obs_num}', 2) # shouldn't be this single?
            self.obs_origin_par_list.append(obs_origin_par)

            obs_weight_par = self.prb.createParameter(f'obs_weight_{obs_num}', 1)
            self.obs_weight_par_list.append(obs_weight_par)

            obs_fun = CasadiObstacle().simpleFormulation()(grid_origin, obs_origin_par, radius_robot, self.obstacle_radius)
            self.f_obs_grid += obs_weight_par * utils.utils.barrier(obs_fun)

        self.prb.createResidual('obstacle_grid', self.f_obs_grid)

    def __init_markers(self):

        # publish robot sphere for obstacle avoidance
        self.sphere_marker = Marker()
        self.sphere_marker.header.frame_id = "base_link"  # Assuming the frame_id is 'base_link', change as necessary
        self.sphere_marker.header.stamp = rospy.Time.now()
        self.sphere_marker.ns = "sphere"
        self.sphere_marker.id = 0
        self.sphere_marker.type = Marker.SPHERE
        self.sphere_marker.action = Marker.ADD
        self.sphere_marker.pose.position.x = 0  # Adjust as necessary
        self.sphere_marker.pose.position.y = 0  # Adjust as necessary
        self.sphere_marker.pose.position.z = 0  # Adjust as necessary
        self.sphere_marker.pose.orientation.x = 0.0
        self.sphere_marker.pose.orientation.y = 0.0
        self.sphere_marker.pose.orientation.z = 0.0
        self.sphere_marker.pose.orientation.w = 1.0
        self.sphere_marker.scale.x = 2 * self.radius_sphere_robot_x  # diameter
        self.sphere_marker.scale.y = 2 * self.radius_sphere_robot_y  # diameter
        self.sphere_marker.scale.z = 2 * self.radius_sphere_robot_z  # diameter
        self.sphere_marker.color.a = 0.2
        self.sphere_marker.color.r = 0.5
        self.sphere_marker.color.g = 0.0
        self.sphere_marker.color.b = 0.5

    def __init_ros_publisher(self):

        self.robot_pub = rospy.Publisher('robot_marker', Marker, queue_size=10)

    def run(self, solution):

        if self.obs_gen is None:
            raise Exception("obstacle generator not initialized.")

        tic_obstacle = time.time()
        self.obs_gen.run()
        obs_vec = self.obs_gen.getObstacles()

        tic_obstacle_sense = time.time() - tic_obstacle
        print("time to sense obstacles: ", tic_obstacle_sense)

        tic_assign = time.time()
        # reset all obstacle params
        for obs_i_num in range(self.max_obs_num):
            self.obs_weight_par_list[obs_i_num].assign(0.)

        base_pose = self.base_fk(q=solution['q'][:, 0])
        base_pos = base_pose['ee_pos']
        base_rot = base_pose['ee_rot']

        # assign params
        for obs_i_num in range(self.max_obs_num):
            if obs_i_num < len(obs_vec):
                # update obatacle origin with the sensed one. Needs to be transformed into base_link frame
                obs_origin_sensed = base_pos + base_rot @ obs_vec[obs_i_num].getOrigin()

                self.obs_weight_par_list[obs_i_num].assign(self.weight_cost_obs)
                self.obs_origin_par_list[obs_i_num].assign(obs_origin_sensed[:2])

        time_obstacle_assign = time.time() - tic_assign
        print("time to assign values to obstacles: ", time_obstacle_assign)

        self.robot_pub.publish(self.sphere_marker)

        time_obstacles = time.time() - tic_obstacle
        print("time to handle obstacles: ", time_obstacles)
        # self.time_obstacles_list.append(time_obstacles)
