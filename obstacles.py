#!/usr/bin/python3
from heapq import merge

from visualization_msgs.msg import Marker, MarkerArray
import rospy
import numpy as np
import horizon.utils as utils
from horizon_navigation.pyObstacleGenerator import ObstacleGenerator
from horizon_navigation.pyObstacle import CasadiObstacle, SphereObstacle
import time
from dataclasses import dataclass, field
import random
from std_msgs.msg import Float64, Int64, Float64MultiArray, MultiArrayLayout, MultiArrayDimension
import multiprocessing


@dataclass
class ObstacleMapParameters:
    input_topic_name: str
    robot_sphere_radius: np.array  # robot is approximated as a sphere
    robot_sphere_origin: np.array  # origin of sphere approximating the robot w.r.t. the base_link
    occupancy_map_width: float
    occupancy_map_height: float
    occupancy_map_resolution: float
    max_obs_num: int  # maximum number of obstacle allowed
    obstacle_radius: float  # radius of obstacle
    angle_threshold: float  # angle resolution to remove unnecessary obstacles
    min_blind_angle: float  # blindsight of the robot, does not consider obstacles
    max_blind_angle: float
    weight_cost_obs: float  # cost for each obstacle (repulsive force)
    rviz_markers_topic_name: str = "" # name of rviz markers visualization topic
    robot_sphere_publisher_name: str = None  # Will be initialized in __post_init__
    def __post_init__(self):
        # Set greeting based on the 'name' field
        if self.robot_sphere_publisher_name is None:
            self.robot_sphere_publisher_name = self.input_topic_name

    def getParam(self, param_name):
        try:
            # Dynamically access the attribute if it exists
            return getattr(self, param_name)
        except AttributeError:
            # Raise an error if the attribute doesn't exist
            raise AttributeError(f"Parameter '{param_name}' does not exist in ObstacleMapParameters.")

    def setParam(self, param_name, value):
        try:
            # Check if the attribute exists
            if hasattr(self, param_name):
                # Dynamically set the attribute value
                setattr(self, param_name, value)
            else:
                raise AttributeError(f"Parameter '{param_name}' does not exist in ObstacleMapParameters.")
        except AttributeError as e:
            raise e


class ObstacleGeneratorWrapper:

    def __init__(self, prb, model, kin_dyn):

        self.prb = prb
        self.model = model
        self.kin_dyn = kin_dyn

        self.obstacle_generator = dict()
        self.map_parameters = dict()

        self.f_obs_grid = 0  # function of the inputs




        self.map_parameters["velodyne_map"] = ObstacleMapParameters(input_topic_name="/costmap_node/costmap/costmap",
                                                                    robot_sphere_radius=np.array([0.6, 0.6]),
                                                                    robot_sphere_origin=np.matrix([[0.3, -0.3],
                                                                                                   [0.0,  0.0]]),
                                                                    max_obs_num=50,
                                                                    obstacle_radius=0.1,
                                                                    angle_threshold=0.2,
                                                                    min_blind_angle=-np.pi / 6,
                                                                    max_blind_angle=np.pi / 6,
                                                                    occupancy_map_width=6.0,
                                                                    occupancy_map_height=6.0,
                                                                    occupancy_map_resolution=0.01,
                                                                    weight_cost_obs=0.05,
                                                                    robot_sphere_publisher_name="velodyne_map_publisher"
                                                                    )  # 0.001 # 0.0025

        self.sonar_topic_names = ["rr_lat", "rr_sag", "fr_lat", "fr_sag", "fl_lat", "fl_sag", "rl_lat", "rl_sag"]

        for topic_name in self.sonar_topic_names:

            ultrasound_origin = kin_dyn.fk(f'ultrasound_{topic_name}')(q=self.model.q0)['ee_pos'][:2]
            self.map_parameters[topic_name] = ObstacleMapParameters(input_topic_name=f"/sonar_map/{topic_name}",
                                                                    robot_sphere_radius=np.array([.3]),
                                                                    robot_sphere_origin=ultrasound_origin,
                                                                    max_obs_num=10,
                                                                    obstacle_radius=0.05,
                                                                    angle_threshold=0.09,
                                                                    min_blind_angle=-np.pi / 6,
                                                                    max_blind_angle=np.pi / 6,
                                                                    occupancy_map_width=2.0,
                                                                    occupancy_map_height=2.0,
                                                                    occupancy_map_resolution=0.01,
                                                                    weight_cost_obs=0.05,
                                                                    rviz_markers_topic_name="sonar_map/obstacles",
                                                                    robot_sphere_publisher_name="sonar_map_publisher"
                                                                    )  # 0.001 # 0.0025

        self.obs_origin_par_dict = dict()  # dict of origin parameters for each obstacle
        self.obs_weight_par_dict = dict()  # dict of weight parameters for each obstacle
        self.obs_radius_par_dict = dict()  # dict of radius parameters for each obstacle
        self.robot_sphere_par_dict = dict() # dict of radius parameters for each sphere approximation of robot

        # self.obstacle_distances = dict()

        self.__init_obstacle_generators()
        self.__init_markers()
        self.__init_ros_publisher()

        self.base_fk = kin_dyn.fk('base_link')

        self.variable_sub_name = []

        self.reconfigurable_topics_name = [("max_obs_num", Int64), ("obstacle_radius", Float64), ("weight_cost_obs", Float64), ("robot_sphere_radius", Float64MultiArray)]

        self.__init_publishers(self.sonar_topic_names)
        self.__init_subscribers(self.map_parameters.keys(), self.reconfigurable_topics_name)

        self.variable_change_flag = dict()
        for suffix in self.map_parameters.keys():
            self.variable_change_flag[suffix] = dict()
            for param_name, param_type in self.reconfigurable_topics_name:
                self.variable_change_flag[suffix][param_name] = False



    def __init_subscribers(self, suffix_names, params):

        self.subscribers = dict()
        for suffix in suffix_names:
            self.subscribers[suffix] = list()

            for param_name, param_type in params:

                topic_name = f'/{suffix}/{param_name}'
                sub = rospy.Subscriber(topic_name, param_type, self.__create_callback(suffix, param_name))
                self.subscribers[suffix].append(sub)
                rospy.loginfo(f"Subscriber for {topic_name} ({param_type}) is ready.")

        # small tapullo for centralizing sonars
        self.subscribers["all_sonars"] = list()
        for param_name, param_type in params:
            sub = rospy.Subscriber(f"all_sonars/{param_name}", param_type, self.__republisher_callback(param_name))
            self.subscribers["all_sonars"].append(sub)


    def __init_publishers(self, sonar_topic_names):
        self.sonar_publishers = dict()
        for (topic, topic_type) in self.reconfigurable_topics_name:
            self.sonar_publishers[topic] = dict()
            for sonar_topic in self.sonar_topic_names:
                self.sonar_publishers[topic][sonar_topic] = rospy.Publisher(f'{sonar_topic}/{topic}', topic_type, queue_size=10)

    def __republisher_callback(self, param_name):

        def republish_callback(msg):

            rospy.loginfo("Received message: %s", msg.data)

            for publisher_name, publisher in self.sonar_publishers[param_name].items():

                publisher.publish(msg)  # Publishing the same data on a different topic

        return republish_callback
    #
    def __create_callback(self, suffix_name, topic_name):

        def update_variables_callback(msg):
            """
            Callback function to update the internal variables based on the received message.
            """

            current_value = msg.data
            last_value = self.map_parameters[suffix_name].getParam(topic_name)

            if isinstance(msg, Float64MultiArray):
                # Update only if the new message is different from the last received one
                if len(current_value) != len(last_value):
                    raise Exception(f'wrong dimension of parameter inserted {len(current_value)} != {len(last_value)}')

                if len(current_value) >= 1:
                    for a, b in zip(current_value, last_value):
                        if a != b:
                            rospy.loginfo(f"Received update of {suffix_name}/{topic_name}: {msg.data}")
                            self.map_parameters[suffix_name].setParam(topic_name, np.array(msg.data))

            else:

                if current_value != last_value:
                    rospy.loginfo(f"Received update of {suffix_name}/{topic_name}: {msg.data}")
                    self.variable_change_flag[suffix_name][topic_name] = True

                    # Update the internal variables
                    self.map_parameters[suffix_name].setParam(topic_name, msg.data)

        return update_variables_callback

    def __init_obstacle_generators(self):

        for layer_name, map_param in self.map_parameters.items():
            self.obstacle_generator[layer_name] = ObstacleGenerator(map_param.occupancy_map_width,
                                                                    map_param.occupancy_map_height,
                                                                    map_param.occupancy_map_resolution,
                                                                    map_param.input_topic_name,
                                                                    map_param.rviz_markers_topic_name
                                                                    )

            self.obstacle_generator[layer_name].setMaxObstacleNum(map_param.max_obs_num)  # maximum number of obstacle allowed
            self.obstacle_generator[layer_name].setObstacleRadius(map_param.obstacle_radius)  # radius of obstacle
            self.obstacle_generator[layer_name].setAngleThreshold(
                map_param.angle_threshold)  # angle resolution to remove unnecessary obstacles (the space is radially divided into sectors of given angle)
            self.obstacle_generator[layer_name].setBlindAngle(map_param.min_blind_angle,
                                                              map_param.max_blind_angle)  # blindsight of the robot, does not consider obstacles

        i_map = 0
        for layer_name, map_param in self.map_parameters.items():

            robot_origin = self.kin_dyn.fk('base_link')(q=self.model.q)['ee_pos'][:2]
            robot_sphere_origin_absolute = robot_origin + map_param.robot_sphere_origin

            # distances from obstacles in each layer are computed from one or more sphere (approximation of the robot)
            # self.obstacle_distances[layer_name] = {f"sphere_{sphere_i}": [] for sphere_i in range(len(map_param.robot_sphere_radius))}

            robot_sphere_radius_par = self.prb.createParameter(f'robot_sphere_radius_map_{i_map}', len(map_param.robot_sphere_radius))
            self.robot_sphere_par_dict[layer_name] = robot_sphere_radius_par

            self.obs_origin_par_dict[layer_name] = list()
            self.obs_weight_par_dict[layer_name] = list()
            self.obs_radius_par_dict[layer_name] = list()

            for obs_num in range(map_param.max_obs_num):
                # add to cost function all the casadi obstacles, parametrized with ORIGIN and WEIGHT
                obs_origin_par = self.prb.createParameter(f'obs_origin_map_{i_map}_{obs_num}', 2)  # shouldn't be this single?
                self.obs_origin_par_dict[layer_name].append(obs_origin_par)

                obs_weight_par = self.prb.createParameter(f'obs_weight_map_{i_map}_{obs_num}', 1)
                self.obs_weight_par_dict[layer_name].append(obs_weight_par)

                obs_radius_par = self.prb.createParameter(f'obs_radius_map_{i_map}_{obs_num}', 1)
                self.obs_radius_par_dict[layer_name].append(obs_radius_par)


                # dictionary of "map layers" with obstacles
                # for each obstacle in a map, distances from all the robot spheres approximation

                for sphere_i in range(len(map_param.robot_sphere_radius)):
                    obs_fun = CasadiObstacle().simpleFormulation()(robot_sphere_origin_absolute[:, sphere_i],
                                                                   obs_origin_par,
                                                                   self.robot_sphere_par_dict[layer_name][sphere_i],
                                                                   obs_radius_par)

                    self.f_obs_grid += obs_weight_par * utils.utils.barrier(obs_fun)
                    # self.obstacle_distances[layer_name][f"sphere_{sphere_i}"].append(obs_fun)

            i_map += 1

        self.prb.createResidual('obstacle_grid', self.f_obs_grid)

    def __init_markers(self):

        self.robot_markers = dict()

        for layer_name, map_param in self.map_parameters.items():

            self.robot_markers[layer_name] = MarkerArray()

            layer_rgb = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]

            for sphere_i in range(len(map_param.robot_sphere_radius)):
                # publish robot sphere for obstacle avoidance
                robot_marker = Marker()
                robot_marker.header.frame_id = "base_link"  # Assuming the frame_id is 'base_link', change as necessary
                robot_marker.header.stamp = rospy.Time.now()
                robot_marker.ns = layer_name
                robot_marker.id = sphere_i
                robot_marker.type = Marker.SPHERE
                robot_marker.action = Marker.ADD
                robot_marker.pose.position.x = map_param.robot_sphere_origin[0, sphere_i]  # Adjust as necessary
                robot_marker.pose.position.y = map_param.robot_sphere_origin[1, sphere_i]  # Adjust as necessary
                robot_marker.pose.position.z = 0  # Adjust as necessary
                robot_marker.pose.orientation.x = 0.0
                robot_marker.pose.orientation.y = 0.0
                robot_marker.pose.orientation.z = 0.0
                robot_marker.pose.orientation.w = 1.0
                robot_marker.scale.x = 2 * map_param.robot_sphere_radius[sphere_i]  # diameter
                robot_marker.scale.y = 2 * map_param.robot_sphere_radius[sphere_i]  # diameter
                robot_marker.scale.z = 2 * 0.01  # diameter
                robot_marker.color.a = 0.1
                robot_marker.color.r = layer_rgb[0]
                robot_marker.color.g = layer_rgb[1]
                robot_marker.color.b = layer_rgb[2]

                self.robot_markers[layer_name].markers.append(robot_marker)

    def __update_markers(self):

        for layer_name, layer_info in self.map_parameters.items():
            for sphere_i in range(len(layer_info.robot_sphere_radius)):

                self.robot_markers[layer_name].markers[sphere_i].scale.x = 2 * layer_info.robot_sphere_radius[sphere_i]  # diameter
                self.robot_markers[layer_name].markers[sphere_i].scale.y = 2 * layer_info.robot_sphere_radius[sphere_i]  # diameter


    def __init_ros_publisher(self):
        self.robot_pub = dict()
        for _, layer_info in self.map_parameters.items():
            if layer_info.robot_sphere_publisher_name not in self.robot_pub:
                self.robot_pub[layer_info.robot_sphere_publisher_name] = rospy.Publisher(f'{layer_info.robot_sphere_publisher_name}/robot_markers', MarkerArray, queue_size=10)

    def run(self, solution):

        if self.obstacle_generator is None:
            raise Exception("obstacle generator not initialized.")

        tic_obstacle = time.time()
        obs_vec = dict()

        for layer_name, obstacle_layer in self.obstacle_generator.items():
            if self.variable_change_flag[layer_name]['max_obs_num']:
                obstacle_layer.setMaxObstacleNum(self.map_parameters[layer_name].max_obs_num)  # maximum number of obstacle allowed
                self.variable_change_flag[layer_name]['max_obs_num'] = False
            if self.variable_change_flag[layer_name]['obstacle_radius']:
                obstacle_layer.setObstacleRadius(self.map_parameters[layer_name].obstacle_radius)  # radius of obstacle
                self.variable_change_flag[layer_name]['obstacle_radius'] = False


        for generator_name, generator in self.obstacle_generator.items():
            # tic_single_generator_time = time.time()
            generator.run()
            # toc_single_generator_time = time.time() - tic_single_generator_time
            # print("time to run single generator: ", toc_single_generator_time)
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

            self.robot_sphere_par_dict[map_name].assign(map_params.robot_sphere_radius)

            for obs_i_num in range(map_params.max_obs_num):
                if obs_i_num < len(obs_vec[map_name]):
                    # update obatacle origin with the sensed one. Needs to be transformed into base_link frame
                    obs_origin_sensed = base_pos + base_rot @ obs_vec[map_name][obs_i_num].getOrigin()

                    self.obs_radius_par_dict[map_name][obs_i_num].assign(map_params.obstacle_radius)
                    self.obs_weight_par_dict[map_name][obs_i_num].assign(map_params.weight_cost_obs)
                    self.obs_origin_par_dict[map_name][obs_i_num].assign(obs_origin_sensed[:2])

        time_obstacle_assign = time.time() - tic_assign
        print("time to assign values to obstacles: ", time_obstacle_assign)

        self.__update_markers()

        # ====================================================================
        merged_array = dict()
        for publisher_name in self.robot_pub:
            merged_array[publisher_name] = MarkerArray()


        for layer_name, marker_array in self.robot_markers.items():
            merged_array[self.map_parameters[layer_name].robot_sphere_publisher_name].markers += marker_array.markers

        for publisher_name in self.robot_pub:
            self.robot_pub[publisher_name].publish(merged_array[publisher_name])

        time_obstacles = time.time() - tic_obstacle
        print("time to handle obstacles: ", time_obstacles)
        # self.time_obstacles_list.append(time_obstacles)

    # def getObstacleDistances(self):
    #     return self.obstacle_distances

    def getObstacleWeightParameter(self):
        return self.obs_weight_par_dict
