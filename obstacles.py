#!/usr/bin/python3

from visualization_msgs.msg import Marker, MarkerArray
import rospy
import numpy as np
import horizon.utils as utils
from horizon_navigation.pyObstacleGenerator import ObstacleGenerator
from horizon_navigation.pyObstacle import CasadiObstacle, SphereObstacle
import time
import rospkg
from dataclasses import dataclass, field


@dataclass
class ObstacleMapParameters:
    occupancy_map_width: float
    occupancy_map_height: float
    occupancy_map_resolution: float
    max_obs_num: int  # maximum number of obstacle allowed
    obstacle_radius: float  # radius of obstacle
    angle_threshold: float  # angle resolution to remove unnecessary obstacles
    min_blind_angle: float  # blindsight of the robot, does not consider obstacles
    max_blind_angle: float
    weight_cost_obs: float  # cost for each obstacle (repulsive force)


class ObstacleGeneratorWrapper:

    def __init__(self, prb, model, kin_dyn):

        self.prb = prb
        self.model = model
        self.kin_dyn = kin_dyn

        self.obstacle_generator = dict()
        self.map_parameters = dict()

        # robot is approximated as a sphere
        self.radius_spheres_robot = [.5, .5]
                                              #     x            y
        self.origin_spheres_robot = np.matrix([[0.3, -0.3], [0.0, 0.0]])  # origin of sphere approximating the robot w.r.t. the base_link
        # self.radius_sphere_robot_y = 0.8

        self.f_obs_grid = 0  # function of the inputs

        self.map_parameters["costmap_node/costmap/costmap"] = ObstacleMapParameters(max_obs_num=50,
                                                                                    obstacle_radius=0.1,
                                                                                    angle_threshold=0.2,
                                                                                    min_blind_angle=-np.pi / 6,
                                                                                    max_blind_angle=np.pi / 6,
                                                                                    occupancy_map_width=6.0,
                                                                                    occupancy_map_height=6.0,
                                                                                    occupancy_map_resolution=0.01,
                                                                                    weight_cost_obs=0.02)  # 0.001 # 0.0025

        self.map_parameters["sonar_map"] = ObstacleMapParameters(max_obs_num=20,
                                                                 obstacle_radius=0.05,
                                                                 angle_threshold=0.09,
                                                                 min_blind_angle=-np.pi / 6,
                                                                 max_blind_angle=np.pi / 6,
                                                                 occupancy_map_width=6.0,
                                                                 occupancy_map_height=6.0,
                                                                 occupancy_map_resolution=0.01,
                                                                 weight_cost_obs=0.01)  # 0.001 # 0.0025

        self.obs_origin_par_dict = dict()  # dict of origin parameters for each obstacle
        self.obs_weight_par_dict = dict()  # dict of weight parameters for each obstacle

        self.obstacle_distances = dict()

        # self.time_obstacles_list = list()

        self.__init_obstacle_generators()
        self.__init_markers()
        self.__init_ros_publisher()

        self.base_fk = kin_dyn.fk('base_link')

    def __init_obstacle_generators(self):

        for layer_name, map_param in self.map_parameters.items():
            self.obstacle_generator[layer_name] = ObstacleGenerator(map_param.occupancy_map_width,
                                                                    map_param.occupancy_map_height,
                                                                    map_param.occupancy_map_resolution,
                                                                    [f"/{layer_name}"]
                                                                    )

            self.obstacle_generator[layer_name].setMaxObstacleNum(map_param.max_obs_num)  # maximum number of obstacle allowed
            self.obstacle_generator[layer_name].setObstacleRadius(map_param.obstacle_radius)  # radius of obstacle
            self.obstacle_generator[layer_name].setAngleThreshold(map_param.angle_threshold)  # angle resolution to remove unnecessary obstacles (the space is radially divided into sectors of given angle)
            self.obstacle_generator[layer_name].setBlindAngle(map_param.min_blind_angle,
                                                              map_param.max_blind_angle)  # blindsight of the robot, does not consider obstacles

        robot_origin = self.kin_dyn.fk('base_link')(q=self.model.q)['ee_pos'][:2]
        self.robot_sphere_origin_absolute = robot_origin + self.origin_spheres_robot

        i_map = 0
        for layer_name, map_param in self.map_parameters.items():

            # distances from obstacles in each layer are computed from one or more sphere (approximation of the robot)
            self.obstacle_distances[layer_name] = {f"sphere_{sphere_i}": [] for sphere_i in range(len(self.radius_spheres_robot))}
            self.obs_origin_par_dict[layer_name] = list()
            self.obs_weight_par_dict[layer_name] = list()

            for obs_num in range(map_param.max_obs_num):
                # add to cost function all the casadi obstacles, parametrized with ORIGIN and WEIGHT
                obs_origin_par = self.prb.createParameter(f'obs_origin_map_{i_map}_{obs_num}', 2)  # shouldn't be this single?
                self.obs_origin_par_dict[layer_name].append(obs_origin_par)

                obs_weight_par = self.prb.createParameter(f'obs_weight_map_{i_map}_{obs_num}', 1)
                self.obs_weight_par_dict[layer_name].append(obs_weight_par)

                # obs_fun = CasadiObstacle().simpleFormulation()(robot_sphere_origin, obs_origin_par, self.radius_sphere_robot, map_param.obstacle_radius)
                # self.f_obs_grid += obs_weight_par * utils.utils.barrier(obs_fun)
                # self.obstacle_distances[map_name].append(obs_fun)

                # dictionary of "map layers" with obstacles
                # for each obstacle in a map, distances from all the robot spheres approximation
                for sphere_i in range(len(self.radius_spheres_robot)):

                    obs_fun = CasadiObstacle().simpleFormulation()(self.robot_sphere_origin_absolute[:, sphere_i],
                                                                   obs_origin_par,
                                                                   self.radius_spheres_robot[sphere_i],
                                                                   map_param.obstacle_radius)

                    self.f_obs_grid += obs_weight_par * utils.utils.barrier(obs_fun)
                    self.obstacle_distances[layer_name][f"sphere_{sphere_i}"].append(obs_fun)

            i_map += 1

        self.prb.createResidual('obstacle_grid', self.f_obs_grid)

    def __init_markers(self):

        self.robot_markers = MarkerArray()
        for sphere_i in range(len(self.radius_spheres_robot)):
            # publish robot sphere for obstacle avoidance
            robot_marker = Marker()
            robot_marker.header.frame_id = "base_link"  # Assuming the frame_id is 'base_link', change as necessary
            robot_marker.header.stamp = rospy.Time.now()
            robot_marker.ns = "sphere"
            robot_marker.id = sphere_i
            robot_marker.type = Marker.SPHERE
            robot_marker.action = Marker.ADD
            robot_marker.pose.position.x = self.origin_spheres_robot[0, sphere_i]  # Adjust as necessary
            robot_marker.pose.position.y = self.origin_spheres_robot[1, sphere_i]  # Adjust as necessary
            robot_marker.pose.position.z = 0  # Adjust as necessary
            robot_marker.pose.orientation.x = 0.0
            robot_marker.pose.orientation.y = 0.0
            robot_marker.pose.orientation.z = 0.0
            robot_marker.pose.orientation.w = 1.0
            robot_marker.scale.x = 2 * self.radius_spheres_robot[sphere_i]  # diameter
            robot_marker.scale.y = 2 * self.radius_spheres_robot[sphere_i]  # diameter
            robot_marker.scale.z = 2 * 0.01  # diameter
            robot_marker.color.a = 0.2
            robot_marker.color.r = 0.5
            robot_marker.color.g = 0.0
            robot_marker.color.b = 0.5

            self.robot_markers.markers.append(robot_marker)

    def __init_ros_publisher(self):

        self.robot_pub = rospy.Publisher('robot_marker', MarkerArray, queue_size=10)

    def run(self, solution):

        if self.obstacle_generator is None:
            raise Exception("obstacle generator not initialized.")

        tic_obstacle = time.time()
        obs_vec = dict()
        for generator_name, generator in self.obstacle_generator.items():
            generator.run()
            obs_vec[generator_name] = generator.getObstacles()

        tic_obstacle_sense = time.time() - tic_obstacle
        print("time to sense obstacles: ", tic_obstacle_sense)

        tic_assign = time.time()

        # reset all obstacle params for all map
        for _, obs_weight_list in self.obs_weight_par_dict.items():
            for obs_weight in obs_weight_list:
                obs_weight.assign(0.)

        base_pose = self.base_fk(q=solution['q'][:, 0])
        base_pos = base_pose['ee_pos']
        base_rot = base_pose['ee_rot']

        # assign params
        for map_name, map_params in self.map_parameters.items():
            for obs_i_num in range(map_params.max_obs_num):
                if obs_i_num < len(obs_vec[map_name]):
                    # update obatacle origin with the sensed one. Needs to be transformed into base_link frame
                    obs_origin_sensed = base_pos + base_rot @ obs_vec[map_name][obs_i_num].getOrigin()

                    self.obs_weight_par_dict[map_name][obs_i_num].assign(map_params.weight_cost_obs)
                    self.obs_origin_par_dict[map_name][obs_i_num].assign(obs_origin_sensed[:2])

        time_obstacle_assign = time.time() - tic_assign
        print("time to assign values to obstacles: ", time_obstacle_assign)

        self.robot_pub.publish(self.robot_markers)

        time_obstacles = time.time() - tic_obstacle
        print("time to handle obstacles: ", time_obstacles)
        # self.time_obstacles_list.append(time_obstacles)

    def getObstacleDistances(self):
        return self.obstacle_distances

    def getObstacleWeightParameter(self):
        return self.obs_weight_par_dict
