#!/usr/bin/python3

from std_srvs.srv import SetBool
import tf2_ros
from geometry_msgs.msg import TransformStamped, WrenchStamped
from visualization_msgs.msg import Marker
import rospy
import numpy as np
import copy
from horizon.rhc import taskInterface
from scipy.spatial.transform import Rotation

from force_joystick import ForceJoystick
from joy_commands import JoyForce

from modes import OperationMode


class VirtualMassHandler:
    def __init__(self, kin_dyn, initial_solution, ti: taskInterface, input_mode='sensor'):

        self.kin_dyn = kin_dyn

        self.dt = ti.prb.getDt()
        self.ns = ti.prb.getNNodes()

        self.__base_yaw_control_flag = True

        # m_virtual = np.array([50, 50])
        # k_virtual = np.array([50, 50])
        # d_virtual = np.array([20, 20])

        # expose this outside
        self.m_virtual = np.array([50, 50, 50]) # 80 80 80 slow but good
        self.k_virtual = np.array([0, 0, 0])
        self.d_virtual = np.array([50, 50, 50]) # 70 70 70 slow but good

        # critical damping
        # 2 * np.sqrt(k_virtual[0] * m_virtual[0]
        # 2 * np.sqrt(k_virtual[1] * m_virtual[1]
        self.solution = initial_solution

        self.ee_name = 'ee_E'

        self.virtual_mass_controller = self.__init_virtual_mass_controller()
        self.sys_dim = self.virtual_mass_controller.getDimension()

        # ee task
        self.ee_task_name = 'ee_force'
        self.ee_task = ti.getTask(self.ee_task_name)

        ## posture task
        self.posture_arm_name = 'posture_arm'
        self.posture_arm_task = ti.getTask(self.posture_arm_name)
        self.initial_w_posture_arm = self.posture_arm_task.getWeight()


        ## required for omnisteering
        # floating base task
        self.posture_cart_name = "posture_base"
        self.posture_cart_task = ti.getTask(self.posture_cart_name)

        # kin fun of end effector
        self.ee_fk_pose_fun = kin_dyn.fk(self.ee_name)
        self.ee_fk_vel_fun = kin_dyn.frameVelocity(self.ee_name, ti.model.kd_frame)

        # get pose and linear+angular velocity
        self.ee_initial_pose = self.ee_fk_pose_fun(q=self.solution['q'][:, 0])
        self.ee_initial_vel = self.ee_fk_vel_fun(q=self.solution['q'][:, 0], qdot=self.solution['v'][:, 0])

        # get position and linear velocity
        self.ee_initial_pos = copy.copy(self.ee_initial_pose['ee_pos'][:self.sys_dim].full())
        self.ee_initial_vel_lin = copy.copy(self.ee_initial_vel['ee_vel_linear'][:self.sys_dim].full())


        if self.__base_yaw_control_flag:

            ## base task
            self.base_force_name = 'base_force'
            self.base_force_task = ti.getTask(self.base_force_name)

            # kin_dyn functions of base
            self.base_fk_pose_fun = kin_dyn.fk('base_link')
            self.base_fk_vel_fun = kin_dyn.frameVelocity('base_link', ti.model.kd_frame)

            # get yaw angle of base
            self.base_initial_rot = self.base_fk_pose_fun(q=self.solution['q'][:, 0])['ee_rot'] # matrix
            self.base_initial_yaw = Rotation.from_matrix(self.base_initial_rot).as_euler("xyz")[2] # yaw angle

            # get yaw velocity of base
            self.base_initial_yaw_vel = self.base_fk_vel_fun(q=self.solution['q'][:, 0], qdot=self.solution['v'][:, 0])['ee_vel_angular'].full()[2]

            # virtual mass controller initialized with x_ee-y_ee, yaw_base
            self.ee_xy_base_yaw = np.array([[self.ee_initial_pos[0, 0], self.ee_initial_pos[1, 0], self.base_initial_yaw]]).T
            self.ee_xy_base_yaw_vel = np.array([[self.ee_initial_vel_lin[0, 0], self.ee_initial_vel_lin[1, 0], self.base_initial_yaw_vel[0]]]).T

            # set initial pose
            self.virtual_mass_controller.setPositionReference(self.ee_xy_base_yaw)

            # set initial state
            self.ee_integrated = np.vstack([self.ee_xy_base_yaw, self.ee_xy_base_yaw_vel])

            # get reference of base task force
            self.base_ref = self.base_force_task.getValues()

        else:

            # set initial pose
            self.virtual_mass_controller.setPositionReference(self.ee_initial_pos)

            # set initial state
            self.ee_integrated = np.vstack([self.ee_initial_pos, self.ee_initial_vel_lin])

        # get reference of ee task force
        self.ee_wrench = np.zeros(6)
        self.ee_ref = self.ee_task.getValues()
        self.ee_ref[3:7, :] = np.array([[0, 0, 0, 1]]).T

        self.ee_homing_posture = copy.copy(self.solution['q'][15:22, :])
        # ee z task
        # self.ee_z_task = ti.getTask('ee_z_force')

        # ===============================================

        self.input_mode = input_mode  # 'joystick' 'sensor'
        self.operation_mode = OperationMode.IDLE

        if self.input_mode == 'joystick':
            self.__init_joystick()
        elif self.input_mode == 'sensor':
            self.__init_subscribers()

        # compute initial wrench offset
        wrench_init_rate = rospy.Rate(500)
        for i in range(50):
            self.wrench_offset = self.ee_wrench
            wrench_init_rate.sleep()
            i += 1

        print(f'Wrench offset: {self.wrench_offset}')

        self.__init_publisher()
        self.__init_services()

    def __init_publisher(self):

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.transform_ref = TransformStamped()

        self.marker_pub = rospy.Publisher('force_marker', Marker, queue_size=10)

        self.marker_ref = Marker()
        self.marker_ref.header.frame_id = "world"  # Set your desired frame ID
        self.marker_ref.id = 0
        self.marker_ref.type = Marker.ARROW
        self.marker_ref.action = Marker.ADD

        # Set the scale of the marker
        self.marker_ref.scale.x = 0.2
        self.marker_ref.scale.y = 0.02
        self.marker_ref.scale.z = 0.02

        # Set the color of the marker
        self.marker_ref.color.r = 1.0
        self.marker_ref.color.g = 0.0
        self.marker_ref.color.b = 0.0
        self.marker_ref.color.a = 1.0  # Fully opaque

    def __capture_homing(self, req):
        if req.data:
            self.ee_homing_posture = copy.copy(self.solution['q'][15:22, :])

        return {'success': True}

    def __init_services(self):

        print('Opening services for virtual mass handler...\n')
        # teaching mode
        self.follow_me_mode_service = rospy.Service('/force_mpc/capture_homing/switch', SetBool, self.__capture_homing)

        print("done.\n")

    def __init_joystick(self):
        self.jc = JoyForce()

    def __init_subscribers(self):
        print('Subscribing to force estimation topic...')
        rospy.Subscriber('/force_estimation/local', WrenchStamped, self.__wrench_callback)  # /cartesian/force_estimation/ee_E
        print("done.")

    def __init_virtual_mass_controller(self):

        sys_dim = 3

        vmass_opt = dict(mass=self.m_virtual, damp=self.d_virtual, spring=self.k_virtual)
        return ForceJoystick(dt=self.dt, n_step=self.ns, sys_dim=sys_dim, opt=vmass_opt)

    def __wrench_callback(self, msg):
        self.ee_wrench = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                                   msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])

    def __integrate(self, q_current, qdot_current, ee_wrench_sensed, wrench_local_frame=False):

        # get wrench
        force_sensed = ee_wrench_sensed[:3]

        # get current position of the ee on xy
        ee_pose = self.ee_fk_pose_fun(q=q_current)
        ee_vel = self.ee_fk_vel_fun(q=q_current, qdot=qdot_current)

        ee_pos = ee_pose['ee_pos'][:self.sys_dim].full()
        ee_vel_lin = ee_vel['ee_vel_linear'][:self.sys_dim]

        # controller works in world frame
        if self.input_mode == 'sensor' and wrench_local_frame:
            # rotate in world from local ee
            ee_rot = ee_pose['ee_rot']
            force_sensed_rot = (ee_rot @ force_sensed)[:self.sys_dim]
        else:
            # force in world coordinates from joystick
            force_sensed_rot = force_sensed

        # ignore z if follow me is on
        if self.operation_mode == OperationMode.FOLLOW_ME:
            force_sensed_rot[2] = 0.0
            # force_sensed_rot[2] = copy.copy(force_sensed_rot[1])

        # compute virtual mass displacement

        # with integrated state
        if self.operation_mode == OperationMode.HYBRID:
            # uses world coordinates
            self.virtual_mass_controller.update(self.ee_integrated[:, 0], force_sensed_rot[:self.sys_dim])
        else:

            # with real state
            if self.__base_yaw_control_flag:

                # get current yaw angle of the base
                base_pose = self.base_fk_pose_fun(q=q_current)
                base_vel = self.base_fk_vel_fun(q=q_current, qdot=qdot_current)


                base_yaw = Rotation.from_matrix(base_pose['ee_rot']).as_euler("xyz")[2]
                base_yaw_vel = base_vel['ee_vel_angular'].full()[2]

                # cross product between force sensed (in world) and vector rotated as the base_link
                force_sensed_rot[2] = np.cross(np.array(base_pose['ee_rot']) @ np.array([[1, 0, 0]]).T, force_sensed_rot.reshape((3, 1)), axis=0)[2]

                # using xy of ee and yaw of base
                ee_x_base_yaw = np.zeros([3, 1])
                ee_x_base_yaw[0] = ee_pos[0][0]
                ee_x_base_yaw[1] = ee_pos[1][0]
                ee_x_base_yaw[2] = base_yaw

                ee_x_base_yaw_vel = np.zeros([3, 1])
                ee_x_base_yaw_vel[0] = ee_vel_lin[0][0]
                ee_x_base_yaw_vel[1] = ee_vel_lin[1][0]
                ee_x_base_yaw_vel[2] = base_yaw_vel

                self.virtual_mass_controller.update(np.vstack([ee_x_base_yaw, ee_x_base_yaw_vel]), force_sensed_rot[:self.sys_dim])

            else:
                # using xyz of ee
                self.virtual_mass_controller.update(np.vstack([ee_pos, ee_vel_lin]), force_sensed_rot[:self.sys_dim])

        self.ee_integrated = self.virtual_mass_controller.getIntegratedState()

        return self.ee_integrated

    def setMode(self, mode):

        print("setting operation mode: ", mode)

        if mode == OperationMode.TEACH:

            # activate ee task
            self.ee_task.setWeight(1.)
            # remove postural of arm
            self.posture_arm_task.setWeight(0.0)

            # only for OMNISTEERING
            # ref = np.atleast_2d(solution['q'][:7, 0]).T
            ref = np.array([[0, 0, 0, 0, 0, 0]]).T  # ref in velocity
            self.posture_cart_task.setRef(ref)
            self.posture_cart_task.setWeight(100.)

            self.operation_mode = OperationMode.TEACH

        elif mode == OperationMode.FOLLOW_ME:

            # activate ee task
            self.ee_task.setWeight(1.0)

            if self.__base_yaw_control_flag:
                self.base_force_task.setWeight(0.1)

            # only for OMNISTEERING
            self.posture_cart_task.setWeight(0.)

            # self.posture_arm_task.setRef(self.solution['q'][7:13, :])
            self.posture_arm_task.setRef(self.solution['q'][15:22, :])  # saving the current position of the arm
            self.posture_arm_task.setWeight(1.0)
            self.operation_mode = OperationMode.FOLLOW_ME

        elif mode == OperationMode.HYBRID:

            # activate ee task
            self.ee_task.setWeight(1.0)

            # only for OMNISTEERING
            self.posture_cart_task.setWeight(0.)  # in velocity

            # self.posture_arm_task.setRef(self.solution['q'][7:13, :])
            self.posture_arm_task.setRef(self.solution['q'][15:22, :])  # saving the current position of the arm
            self.posture_arm_task.setWeight(0.1)

            self.operation_mode = OperationMode.HYBRID

        elif mode == OperationMode.HOMING:

            # activate ee task
            self.ee_task.setWeight(0.)

            # only for OMNISTEERING
            self.posture_cart_task.setWeight(1.0)

            self.posture_arm_task.setRef(self.ee_homing_posture)
            self.posture_arm_task.setWeight(0.004)  # set how fast it goes to homing position

            self.operation_mode = OperationMode.HOMING

        elif mode == OperationMode.IDLE:

            self.ee_task.setWeight(0.0)

            if self.__base_yaw_control_flag:
                self.base_force_task.setWeight(0.0)

            self.posture_arm_task.setRef(self.solution['q'][15:22, :])  # saving the current position of the arm
            self.posture_arm_task.setWeight(self.initial_w_posture_arm)
            
            self.operation_mode = OperationMode.IDLE

        else:
            raise Exception('Mode not recognized.')

    def getMode(self):
        return self.operation_mode

    def publish_tf(self, ref):

        self.transform_ref.header.stamp = rospy.Time.now()
        self.transform_ref.header.frame_id = 'world'
        self.transform_ref.child_frame_id = 'force_ref'
        self.transform_ref.transform.translation.x = ref[0, 0]
        self.transform_ref.transform.translation.y = ref[1, 0]
        self.transform_ref.transform.translation.z = ref[2, 0]
        self.transform_ref.transform.rotation.x = ref[3, 0]
        self.transform_ref.transform.rotation.y = ref[4, 0]
        self.transform_ref.transform.rotation.z = ref[5, 0]
        self.transform_ref.transform.rotation.w = ref[6, 0]

        self.tf_broadcaster.sendTransform(self.transform_ref)

    def publish_marker(self, ref):

        self.marker_ref.header.stamp = rospy.Time.now()
        self.marker_ref.pose.position.x = ref[0, 0]
        self.marker_ref.pose.position.y = ref[1, 0]
        self.marker_ref.pose.position.z = ref[2, 0]
        self.marker_ref.pose.orientation.x = ref[3, 0]
        self.marker_ref.pose.orientation.y = ref[4, 0]
        self.marker_ref.pose.orientation.z = ref[5, 0]
        self.marker_ref.pose.orientation.w = ref[6, 0]

        self.marker_pub.publish(self.marker_ref)

    def run(self, solution):

        self.solution = solution

        # select input mode
        if self.input_mode == 'joystick':
            self.jc.run(self.solution)
            force_sensed = self.jc.getForce().T
        elif self.input_mode == 'sensor':
            force_sensed = self.ee_wrench - self.wrench_offset

        else:
            raise Exception('Wrong input mode')

        # get reference
        self.__integrate(self.solution['q'][:, 0],
                         self.solution['v'][:, 0],
                         force_sensed,
                         wrench_local_frame=True)

        self.ee_ref[:2, :] = self.ee_integrated[:2, :]

        if self.__base_yaw_control_flag:
            # using xyz of ee

            self.base_ref[3:7, :] = Rotation.from_euler('z', self.ee_integrated[2, :]).as_quat().T



        else:
            # using xy of ee and yaw of base
            self.ee_ref[:self.sys_dim, :] = self.ee_integrated[:self.sys_dim, :]

        if self.operation_mode != OperationMode.IDLE:
            self.ee_task.setRef(self.ee_ref)
            if self.__base_yaw_control_flag:
                self.base_force_task.setRef(self.base_ref)
            # self.ee_z_task.setRef(self.ee_ref)

        # self.publish_tf(self.ee_ref)
        # self.publish_marker(self.ee_ref)
