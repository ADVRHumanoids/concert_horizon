#!/usr/bin/python3

from horizon.problem import Problem
from horizon.rhc.model_description import FullModelInverseDynamics
from horizon.rhc.taskInterface import TaskInterface
from horizon.utils import utils
from horizon.ros import replay_trajectory
import casadi_kin_dyn.py3casadi_kin_dyn as casadi_kin_dyn
from xbot_interface import config_options as co
from xbot_interface import xbot_interface as xbot
from geometry_msgs.msg import Vector3
from kyon_controller.msg import WBTrajectory
from geometry_msgs.msg import Twist
import casadi as cs
import rospy
import rospkg
import numpy as np
import subprocess
import os
import time
import horizon.utils as utils
from cartesian_interface.pyci_all import *
from mode_handler import ModeHandler
from virtual_mass_handler import VirtualMassHandler
from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init
from scipy.spatial.transform import Rotation as scipy_rot
import copy


xbot_mode = True
obstacle_avoidance = True

rospy.init_node('concert_receding')

virtual_mass_input_mode = rospy.get_param('~input_mode', 'sensor')

if obstacle_avoidance:
    roscpp_init('concert_obstacles', [])

solution_publisher = rospy.Publisher('/mpc_solution', WBTrajectory, queue_size=1, tcp_nodelay=True)
rospy.sleep(1.)

# get from ros param the urdf and srdf
urdf = rospy.get_param(param_name='/robot_description', default='')
if urdf == '':
    raise print('urdf not set')

srdf = rospy.get_param(param_name='/robot_description_semantic', default='')
if srdf == '':
    raise print('srdf not set')

file_dir = os.getcwd()


base_vel_pub = rospy.Publisher('/omnisteering/cmd_vel', Twist, queue_size=10)

'''
Initialize Horizon problem
'''
ns = 30
T = 1.5
dt = T / ns

prb = Problem(ns, receding=True, casadi_type=cs.SX)
prb.setDt(dt)

kin_dyn = casadi_kin_dyn.CasadiKinDyn(urdf)
'''
Build ModelInterface and RobotStatePublisher
'''
cfg = co.ConfigOptions()
cfg.set_urdf(urdf)
cfg.set_srdf(srdf)
cfg.generate_jidmap()
cfg.set_string_parameter('model_type', 'RBDL')
cfg.set_string_parameter('framework', 'ROS')
cfg.set_bool_parameter('is_model_floating_base', True)

robot = None

if xbot_mode:
    print('Getting robot...\n')
    robot = xbot.RobotInterface(cfg)
    robot.sense()
    q_init = robot.getPositionReference()
    q_init = robot.eigenToMap(q_init)
    print('done\n')
else:
    print('XBot-RobotInterface not created.\n Using initial q default values.\n')
    q_init = {'J1_A': 0.0,
              'J_wheel_A': 0.0,
              'J1_B': 0.0,
              'J_wheel_B': 0.0,
              'J1_C': 0.0,
              'J_wheel_C': 0.0,
              'J1_D': 0.0,
              'J_wheel_D': 0.0,
              'J1_E': 0.0,
              'J2_E': -0.5,
              'J3_E': 0.0,
              'J4_E': 0.5,
              'J5_E': 0.0,
              'J6_E': -0.5
              }

# brings to failure
# q_init['J1_A'] = 0.005991784855723381
# q_init['J1_B'] = 0.00839337520301342
# q_init['J1_C'] = -0.0031701645348221064
# q_init['J1_D'] = -0.0030719360802322626

# this is a problem: if not starting at exactly zero, the solver won't move the base
# q_init['J1_A'] = 0
# q_init['J1_B'] = 0
# q_init['J1_C'] = 0
# q_init['J1_D'] = 0

base_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


wheel_radius = 0.16
FK = kin_dyn.fk('J_wheel_A')
init_pos_wheel = FK(q=kin_dyn.mapToQ(q_init))['ee_pos']
base_init[2] = -init_pos_wheel[2] + wheel_radius

model = FullModelInverseDynamics(problem=prb,
                                 kd=kin_dyn,
                                 q_init=q_init,
                                 base_init=base_init
                                 )


# create robot description if robot is not found
if not robot:
    rospy.set_param('mpc/robot_description', urdf)
    robot_state_pub_command = 'rosrun robot_state_publisher robot_state_publisher robot_description:=mpc/robot_description'
    robot_state_pub_process = subprocess.Popen(robot_state_pub_command.split(), start_new_session=True)

# spawn the cartesian marker for end effector control
# cartesian_marker_command = 'rosrun cartesian_interface marker_spawner'
# cartesian_marker_process = subprocess.Popen(cartesian_marker_command.split(), start_new_session=True)

ti = TaskInterface(prb=prb, model=model)
ti.setTaskFromYaml(rospkg.RosPack().get_path('concert_horizon') + '/config/concert_config.yaml')

ee_ref = ti.getTask('ee_force').getValues()[:, 0]
ee_pos_0 = kin_dyn.fk('ee_E')(q=model.q0)['ee_pos'][:, 0]
ee_rot_0 = scipy_rot.from_matrix((kin_dyn.fk('ee_E')(q=model.q0)['ee_rot'].full())).as_quat()

print(f"initial ee reference: {ee_ref}")
print(f"initial ee pos: {ee_pos_0}, {ee_rot_0}")

ti.model.q.setBounds(ti.model.q0, ti.model.q0, nodes=0)
# ti.model.v.setBounds(ti.model.v0, ti.model.v0, nodes=0)
# ti.model.a.setBounds(np.zeros([model.a.shape[0], 1]), np.zeros([model.a.shape[0], 1]), nodes=0)
ti.model.q.setInitialGuess(ti.model.q0)
ti.model.v.setInitialGuess(ti.model.v0)


prb.createResidual('max_q', 1e1 * utils.utils.barrier(kin_dyn.q_max()[7:] - model.q[7:]))
prb.createResidual('min_q', 1e1 * utils.utils.barrier1(kin_dyn.q_min()[7:] - model.q[7:]))

vel_lims = model.kd.velocityLimits()
prb.createResidual('max_vel', 1e2 * utils.utils.barrier(vel_lims[7:] - model.v[7:]))
prb.createResidual('min_vel', 1e1 * utils.utils.barrier1(-1 * vel_lims[7:] - model.v[7:]))


# ============== REQUIRED ONLY FOR OMNISTEERING ==============
model.v[2].setBounds(0, 0) # the robot cannot fly
model.v[3:5].setBounds([0, 0], [0,0]) # the robot cannot pitch and roll

wheel_vel_max_index = [elem for elem in kin_dyn.joint_names() if elem not in ['universe', 'reference']].index('J_wheel_A')

vel_lin_max_padding = 0.
vel_ang_max_padding = 0.
base_vel_lin_max = 0.3 #vel_lims[6 + wheel_vel_max_index] * wheel_radius
base_vel_ang_max = 0.3 #(vel_lims[6 + wheel_vel_max_index] * wheel_radius) / 0.6 --> radius of robot more or less

base_vel_lin_max_padded = base_vel_lin_max - vel_lin_max_padding
base_vel_ang_max_padded = base_vel_ang_max - vel_ang_max_padding

print(f"base_vel_lin_max: {base_vel_lin_max_padded} ")
print(f"base_vel_ang_max: {base_vel_ang_max_padded} ")

prb.createResidual('max_base_linear_vel', 1e2 * utils.utils.barrier(base_vel_lin_max_padded - model.v[:2]))
prb.createResidual('min_base_linear_vel', 1e2 * utils.utils.barrier1(- base_vel_lin_max_padded - model.v[:2]))

# base_vel_max
prb.createResidual("max_base_angular_vel", 1e2 * utils.utils.barrier(base_vel_ang_max_padded - model.v[5]))
prb.createResidual("min_base_angular_vel", 1e2 * utils.utils.barrier1(- base_vel_ang_max_padded - model.v[5]))

# ==============================================================

if obstacle_avoidance:
    from obstacles import ObstacleGeneratorWrapper

    ogw = ObstacleGeneratorWrapper(prb, model, kin_dyn)

    # obstacle_distances = ogw.getObstacleDistances()


    # obstacle_distance_function = dict()
    # for layer_name, spheres_robot_dict in obstacle_distances.items():
    #     obstacle_distance_function[layer_name] = dict()
    
    #     for sphere_name, obstacles in spheres_robot_dict.items():
    
    #         obstacle_distance_function[layer_name][sphere_name] = list()
    
    #         for obs_i in range(len(obstacles)):
    
    #             function_name = f"f_{layer_name[:4].replace('/', '')}_{sphere_name}_{obs_i}"
    #             obstacle_distance_function[layer_name][sphere_name].append(prb.createExpr(obstacles[obs_i], function_name))


# finalize taskInterface and solve bootstrap problem
ti.finalize()

ti.bootstrap()
ti.load_initial_guess()
solution = ti.solution

rate = rospy.Rate(1 / dt)

contact_list_repl = list(model.cmap.keys())


repl = replay_trajectory.replay_trajectory(dt, model.kd.joint_names(), np.array([]),
                                           {k: None for k in model.fmap.keys()},
                                           model.kd_frame, model.kd,
                                           trajectory_markers=contact_list_repl)
                                           # future_trajectory_markers={'base_link': 'world', 'J_wheel_D': 'world'})

xig = np.empty([prb.getState().getVars().shape[0], 1])
time_elapsed_shifting_list = list()
time_elapsed_solving_list = list()
time_elapsed_all_list = list()
time_elapsed_obstacles_list = list()

vmc = VirtualMassHandler(kin_dyn, solution, ti, input_mode=virtual_mass_input_mode)

print(f"robot controller starting in mode: {vmc.getMode()}")

# required to switch between nodes
mh = ModeHandler(vmc)

iteration = 0


base_fk = kin_dyn.fk('base_link')
max_iter = 0
while not rospy.is_shutdown(): #and max_iter < 1000:

    max_iter += 1
    # tic = time.time()
    # set initial state and initial guess
    shift_num = -1

    x_opt = solution['x_opt']
    xig = np.roll(x_opt, shift_num, axis=1)
    for i in range(abs(shift_num)):
        xig[:, -1 - i] = x_opt[:, -1]

    prb.getState().setInitialGuess(xig)
    prb.setInitialState(x0=xig[:, 0])

    vmc.run(solution)

    if obstacle_avoidance:
        tic = time.time()
        ogw.run(solution)
        time_elapsed_obstacles = time.time() - tic
        time_elapsed_obstacles_list.append(time_elapsed_obstacles)

    tic = time.time()
    ti.rti()

    # if obstacle_avoidance:
    
    #     obstacle_distances = ogw.getObstacleDistances()
    
    #     obs_dist = list()
    #     for layer_name, obstacles_from_spheres_dict in obstacle_distances.items():
    #         for sphere_name, obstacles in obstacles_from_spheres_dict.items():
    #             for obs_i in range(len(obstacles)):
    #                 if ogw.getObstacleWeightParameter()[layer_name][obs_i].getValues()[0, 0] > 0:
    #                     obs_dist.append(prb.evalExpr(obstacle_distance_function[layer_name][sphere_name][obs_i], ti.solution)[:, 0])
    
    #         if obs_dist:
    #             print(f"layer --> {layer_name} ")
    #             print("       min distance: ", np.min(obs_dist))
                    
    time_elapsed_solving = time.time() - tic
    time_elapsed_solving_list.append(time_elapsed_solving)

    solution = ti.solution

    # integrate with omnisteering - publish the base velocity for omnisteering
    vel_cmd = Twist()
    vel_cmd.linear.x = solution['v'][0, 0]
    vel_cmd.linear.y = solution['v'][1, 0]
    vel_cmd.linear.z = solution['v'][2, 0]

    vel_cmd.angular.x = solution['v'][3, 0]
    vel_cmd.angular.y = solution['v'][4, 0]
    vel_cmd.angular.z = solution['v'][5, 0]

    base_vel_pub.publish(vel_cmd)

    # ================ SOLUTION MESSAGE FOR CONTROLLER =================
    sol_msg = WBTrajectory()
    sol_msg.header.frame_id = 'world'
    sol_msg.header.stamp = rospy.Time.now()

    joint_list = [elem for elem in kin_dyn.joint_names() if elem not in ['universe', 'reference']
                                                                         # 'J1_A', 'J_wheel_A',
                                                                         # 'J1_B', 'J_wheel_B',
                                                                         # 'J1_C', 'J_wheel_C',
                                                                         # 'J1_D', 'J_wheel_D']
                  ]

    sol_msg.joint_names = joint_list

    sol_msg.q = solution['q'][:, 0].tolist()
    sol_msg.v = solution['v'][:, 0].tolist()
    sol_msg.a = solution['a'][:, 0].tolist()

    # sol_msg.q = solution['q'][np.array([0, 1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19]), 0].tolist()
    # sol_msg.v = solution['v'][np.array([0, 1, 2, 3, 4, 5, 14, 15, 16, 17, 18, 19]), 0].tolist()
    # sol_msg.a = solution['a'][np.array([0, 1, 2, 3, 4, 5, 14, 15, 16, 17, 18, 19]), 0].tolist()


    for frame in model.getForceMap():
        sol_msg.force_names.append(frame)
        sol_msg.f.append(
            Vector3(x=solution[f'f_{frame}'][0, 0], y=solution[f'f_{frame}'][1, 0], z=solution[f'f_{frame}'][2, 0]))

    solution_publisher.publish(sol_msg)

    # replay stuff
    if robot is None:
        repl.frame_force_mapping = {cname: solution[f.getName()] for cname, f in ti.model.fmap.items()}
        repl.publish_joints(solution['q'][:, 0])
        # repl.publish_joints(solution['q'][:, ns], prefix='last')
        # repl.publishContactForces(rospy.Time.now(), solution['q'][:, 0], 0)
        # repl.publish_future_trajectory_marker('base_link', solution['q'][0:3, :])
        # repl.publish_future_trajectory_marker('J_wheel_D', solution['q'][8:11, :])

    time_elapsed_all = time.time() - tic
    time_elapsed_all_list.append(time_elapsed_all)

    iteration = iteration + 1
    rate.sleep()

    # print(f"{colorama.Style.RED}MPC loop elapsed time: {time.time() - tic}{colorama.Style.RESET}")

# print(f'average time elapsed shifting: {sum(time_elapsed_shifting_list) / len(time_elapsed_shifting_list)}')
print(f'average time elapsed solving: {sum(time_elapsed_solving_list) / len(time_elapsed_solving_list)}')
print(f'average time obstacles: {sum(time_elapsed_obstacles_list) / len(time_elapsed_obstacles_list)}')
print(f'average time elapsed all: {sum(time_elapsed_all_list) / len(time_elapsed_all_list)}')
