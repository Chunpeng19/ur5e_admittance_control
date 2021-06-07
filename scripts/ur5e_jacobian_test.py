#! /usr/bin/env python
import rospy
import numpy as np
from copy import deepcopy
import time
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import signal

from bandpass_filter import PythonBPF
from filter import PythonHPF

from ur_kinematics.ur_kin_py import forward, forward_link
from kinematics import analytical_ik, nearest_ik_solution

from std_msgs.msg import Float64MultiArray, Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from ur5teleop.msg import jointdata, Joint
from ur_dashboard_msgs.msg import SafetyMode
from ur_dashboard_msgs.srv import IsProgramRunning, GetSafetyMode
from std_msgs.msg import Bool
# Import the module
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics

robot = URDF.from_xml_file("/home/ur5e/ur_ws/src/ur5e_admittance_control/config/ur5e.urdf")

joint_vel_lim = 1.0
sample_rate = 500.0

class ur5e_admittance():
    '''Define joint admittance control for ur5e using endeffector wrench info
    '''
    saftey_mode = -1
    shutdown = False
    enabled = False
    joint_reorder = [2,1,0,3,4,5]
    breaking_stop_time = 0.1 #Stop safely by executing stop in 0.1s

    joint_p_gains_varaible = np.array([5.0, 5.0, 5.0, 10.0, 10.0, 10.0])

    #default_pos = (np.pi/180)*np.array([90.0, -90.0, 90.0, -90.0, -90.0, 180.0])
    default_pos = (np.pi/180)*np.array([90.0, -90.0, 90.0, -90.0, -90.0, 180.0])


    robot_ref_pos = deepcopy(default_pos)

    lower_lims = (np.pi/180)*np.array([0.0, -100.0, 0.0, -180.0, -180.0, 90.0])
    upper_lims = (np.pi/180)*np.array([180.0, 0.0, 175.0, 0.0, 0.0, 270.0])
    conservative_lower_lims = (np.pi/180)*np.array([45.0, -100.0, 45.0, -135.0, -135.0, 135.0])
    conservative_upper_lims = (np.pi/180)*np.array([135, -45.0, 140.0, -45.0, -45.0, 225.0])
    max_joint_speeds = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    max_joint_acc = 2.0 * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    fl = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    fh = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
    fs = sample_rate
    filter = PythonBPF(fs, fl, fh)
    # filter = PythonHPF(fl,fs)

    # define DH-parameters
    DH_alpha = np.pi/2*np.array([0,1,0,0,1,-1])
    DH_a = [0.0,0.0,-0.42500,-0.39225,0.0,0.0]
    DH_d = [0.1625,0.0,0.0,0.1333,0.0997,0.0996]

    # define local cylinder joint inertia matrix
    J1l = np.matrix([[0.0103, 0, 0, 0], [0, 0.0103, 0, 0], [0, 0, 0.0067, 0], [0, 0, 0, 3.7]])
    J2l = np.matrix([[0.1834, 0, 0, 0], [0, 0.6582, 0, 0], [0, 0, 0.4984, 0], [0, 0, 0, 8.393]])
    J3l = np.matrix([[0.0065, 0, 0, 0], [0, 0.1352, 0, 0], [0, 0, 0.1351, 0], [0, 0, 0, 2.275]])
    J4l = np.matrix([[0.0027, 0, 0, 0], [0, 0.0034, 0, 0], [0, 0, 0.0027, 0], [0, 0, 0, 1.219]])
    J5l = np.matrix([[0.0027, 0, 0, 0], [0, 0.0034, 0, 0], [0, 0, 0.0027, 0], [0, 0, 0, 1.219]])
    J6l = np.matrix([[0.00025, 0, 0, 0], [0, 0.00025, 0, 0], [0, 0, 0.00019, 0], [0, 0, 0, 0.1879]])

    inertia_offset = np.array([0.4, 0.4, 0.4, 0.1, 0.1, 0.1])
    # inertia_offset = np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1])

    #define fields that are updated by the subscriber callbacks
    current_joint_positions = np.zeros(6)
    current_joint_velocities = np.zeros(6)

    wrench = np.zeros(6)

    tree = kdl_tree_from_urdf_model(robot)
    print tree.getNrOfSegments()

    # forwawrd kinematics
    chain = tree.getChain("base_link", "wrist_3_link")
    print chain.getNrOfJoints()
    kdl_kin = KDLKinematics(robot, "base_link", "wrist_3_link")

    def __init__(self, test_control_signal = False, conservative_joint_lims = True):
        '''set up controller class variables & parameters'''

        if conservative_joint_lims:
            self.lower_lims = self.conservative_lower_lims
            self.upper_lims = self.conservative_upper_lims

        #launch nodes
        rospy.init_node('admittance_controller', anonymous=True)

        #start robot state subscriber (detects fault or estop press)
        rospy.Subscriber('/ur_hardware_interface/safety_mode', SafetyMode, self.safety_callback)
        #joint feedback subscriber
        rospy.Subscriber("joint_states", JointState, self.joint_state_callback)

        #wrench feedback
        rospy.Subscriber("wrench", WrenchStamped, self.wrench_callback)

        #service to check if robot program is running
        rospy.wait_for_service('/ur_hardware_interface/dashboard/program_running')
        self.remote_control_running = rospy.ServiceProxy('ur_hardware_interface/dashboard/program_running', IsProgramRunning)
        #service to check safety mode
        rospy.wait_for_service('/ur_hardware_interface/dashboard/get_safety_mode')
        self.safety_mode_proxy = rospy.ServiceProxy('/ur_hardware_interface/dashboard/get_safety_mode', GetSafetyMode)

        #start vel publisher
        self.vel_pub = rospy.Publisher("/joint_group_vel_controller/command",
                            Float64MultiArray,
                            queue_size=1)

        self.test_data_pub = rospy.Publisher("/test_data",
                            Float64MultiArray,
                            queue_size=1)

        #set shutdown safety behavior
        rospy.on_shutdown(self.shutdown_safe)
        time.sleep(0.5)
        self.stop_arm() #ensure arm is not moving if it was already

        self.velocity = Float64MultiArray(data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.vel_ref = Float64MultiArray(data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.test_data = Float64MultiArray(data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        print("Joint Limmits: ")
        print(self.upper_lims)
        print(self.lower_lims)

        if not self.ready_to_move():
            print('User action needed before commands can be sent to the robot.')
            self.user_prompt_ready_to_move()
        else:
            print('Ready to move')

    def joint_state_callback(self, data):
        self.current_joint_positions[self.joint_reorder] = data.position
        self.current_joint_velocities[self.joint_reorder] = data.velocity

    def wrench_callback(self, data):
        self.current_wrench = np.array([data.wrench.force.x, data.wrench.force.y, data.wrench.force.z, data.wrench.torque.x, data.wrench.torque.y, data.wrench.torque.z])

    # def wrap_relative_angles(self):
    def safety_callback(self, data):
        '''Detect when safety stop is triggered'''
        self.safety_mode = data.mode
        if not data.mode == 1:
            #estop or protective stop triggered
            #send a breaking command
            print('\nFault Detected, sending stop command\n')
            self.stop_arm() #set commanded velocities to zero
            print('***Please clear the fault and restart the UR-Cap program before continuing***')

    def user_wait_safety_stop(self):
        #wait for user to fix the stop
        while not self.safety_mode == 1:
            raw_input('Safety Stop or other stop condition enabled.\n Correct the fault, then hit enter to continue')

    def ensure_safety_mode(self):
        '''Blocks until the safety mode is 1 (normal)'''
        while not self.safety_mode == 1:
            raw_input('Robot safety mode is not normal, \ncheck the estop and correct any faults, then restart the external control program and hit enter. ')

    def get_safety_mode(self):
        '''Calls get safet mode service, does not return self.safety_mode, which is updated by the safety mode topic, but should be the same.'''
        return self.safety_mode_proxy().safety_mode.mode

    def ready_to_move(self):
        '''returns true if the safety mode is 1 (normal) and the remote program is running'''
        return self.get_safety_mode() == 1 and self.remote_control_running()

    def user_prompt_ready_to_move(self):
        '''Blocking dialog to get the user to reset the safety warnings and start the remote program'''
        while True:
            if not self.get_safety_mode() == 1:
                print(self.get_safety_mode())
                raw_input('Safety mode is not Normal. Please correct the fault, then hit enter.')
            else:
                break
        while True:
            if not self.remote_control_running():
                raw_input('The remote control URCap program has been pause or was not started, please restart it, then hit enter.')
            else:
                break
        print('\nRemote control program is running, and safety mode is Normal\n')

    def is_joint_position(self, position):
        '''Verifies that this is a 1dim numpy array with len 6'''
        if isinstance(position, np.ndarray):
            return position.ndim==1 and len(position)==6
        else:
            return False

    def shutdown_safe(self):
        '''Should ensure that the arm is brought to a stop before exiting'''
        self.shutdown = True
        print('Stopping -> Shutting Down')
        self.stop_arm()
        print('Stopped')
        # self.stop_arm()

    def stop_arm(self, safe = False):
        '''Commands zero velocity until sure the arm is stopped. If safe is False
        commands immediate stop, if set to a positive value, will stop gradually'''

        if safe:
            loop_rate = rospy.Rate(200)
            start_time = time.time()
            start_vel = deepcopy(self.current_joint_velocities)
            max_accel = np.abs(start_vel/self.breaking_stop_time)
            vel_mask = np.ones(6)
            vel_mask[start_vel < 0.0] = -1
            while np.any(np.abs(self.current_joint_velocities)>0.0001) and not rospy.is_shutdown():
                command_vels = [0.0]*6
                loop_time = time.time() - start_time
                for joint in range(len(command_vels)):
                    vel = start_vel[joint] - vel_mask[joint]*max_accel[joint]*loop_time
                    if vel * vel_mask[joint] < 0:
                        vel = 0
                    command_vels[joint] = vel
                self.vel_pub.publish(Float64MultiArray(data = command_vels))
                if np.sum(command_vels) == 0:
                    break
                loop_rate.sleep()

        while np.any(np.abs(self.current_joint_velocities)>0.0001):
            self.vel_pub.publish(Float64MultiArray(data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def in_joint_lims(self, position):
        '''expects an array of joint positions'''
        return np.all(self.lower_lims < position) and np.all(self.upper_lims > position)

    def identify_joint_lim(self, position):
        '''expects an array of joint positions. Prints a human readable list of
        joints that exceed limmits, if any'''
        if self.in_joint_lims(position):
            print("All joints ok")
            return True
        else:
            for i, pos in enumerate(position):
                if pos<self.lower_lims[i]:
                    print('Joint {}: Position {:.5} exceeds lower bound {:.5}'.format(i,pos,self.lower_lims[i]))
                if pos>self.upper_lims[i]:
                    print('Joint {}: Position {:.5} exceeds upper bound {:.5}'.format(i,pos,self.lower_lims[i]))
            return False

    def remote_program_running(self):
        print('remote : ',self.remote_control_running().program_running)
        return self.remote_control_running().program_running

    def move_to_robost(self,
                position,
                speed = 0.25,
                error_thresh = 0.01,
                override_initial_joint_lims = False,
                require_enable = False):
        '''Calls the move_to method as necessary to ensure that the goal position
        is reached, accounting for interruptions due to safety faults, and the
        enable deadman if require_enable is selected'''

        if require_enable:
            print('Depress and hold the deadman switch when ready to move.')
            print('Release to stop')

        while not rospy.is_shutdown():
            #check safety
            if not self.ready_to_move():
                self.user_prompt_ready_to_move()
                continue
            #start moving
            print('Starting Trajectory')
            result = self.move_to(position,
                                  speed = speed,
                                  error_thresh = error_thresh,
                                  override_initial_joint_lims = override_initial_joint_lims,
                                  require_enable = require_enable)
            if result:
                break
        print('Reached Goal')

    def move_to(self,
                position,
                speed = 0.25,
                error_thresh = 0.01,
                override_initial_joint_lims = False,
                require_enable = False):
        '''CAUTION - use joint lim override with extreme caution. Intended to
        allow movement from outside the lims back to acceptable position.

        Defines a simple joing controller to bring the arm to a desired
        configuration without teleop input. Intended for testing or to reach
        present initial positions, etc.'''

        #ensure safety sqitch is not enabled
        if not self.ready_to_move():
            self.user_prompt_ready_to_move()

        #define max speed slow for safety
        if speed > 0.5:
            print("Limiting speed to 0.5 rad/sec")
            speed = 0.5

        #calculate traj from current position
        start_pos = deepcopy(self.current_joint_positions)
        max_disp = np.max(np.abs(position-start_pos))
        end_time = max_disp/speed

        #make sure this is a valid joint position
        if not self.is_joint_position(position):
            print("Invalid Joint Position, Exiting move_to function")
            return False

        #check joint llims
        if not override_initial_joint_lims:
            if not self.identify_joint_lim(start_pos):
                print("Start Position Outside Joint Lims...")
                return False
        if not self.identify_joint_lim(position):
            print("Commanded Postion Outside Joint Lims...")
            return False


        print('Executing Move to : \n{}\nIn {} seconds'.format(position,end_time))
        #list of interpolators ... this is kind of dumb, there is probably a better solution
        traj = [InterpolatedUnivariateSpline([0.,end_time],[start_pos[i],position[i]],k=1) for i in range(6)]

        position_error = np.array([1.0]*6) #set high position error
        pos_ref = deepcopy(start_pos)
        rate = rospy.Rate(500) #lim loop to 500 hz
        start_time = time.time()
        reached_pos = False
        while not self.shutdown and not rospy.is_shutdown() and self.safety_mode == 1: #chutdown is set on ctrl-c.
            if not require_enable:
                print('Lost Enable, stopping')
                break

            loop_time = time.time()-start_time
            if loop_time < end_time:
                pos_ref[:] = [traj[i](loop_time) for i in range(6)]
            else:
                pos_ref = position
                # break
                if np.all(np.abs(position_error)<error_thresh):
                    print("reached target position")
                    self.stop_arm()
                    reached_pos = True
                    break

            position_error = pos_ref - self.current_joint_positions
            vel_ref_temp = self.joint_p_gains_varaible*position_error
            #enforce max velocity setting
            np.clip(vel_ref_temp,-joint_vel_lim,joint_vel_lim,vel_ref_temp)
            self.vel_ref.data = vel_ref_temp
            self.vel_pub.publish(self.vel_ref)
            # print(pos_ref)
            #wait
            rate.sleep()

        #make sure arm stops
        self.stop_arm(safe = True)
        return reached_pos

    def move(self,
             capture_start_as_ref_pos = False,
             dialoge_enabled = True):
        '''Main control loop for teleoperation use.'''
        if not self.ready_to_move():
            self.user_prompt_ready_to_move()

        max_pos_error = 0.5 #radians/sec
        low_joint_vel_lim = 0.5

        zeta = 0.707
        virtual_stiffness = 5.0 * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        joint_torque_error_lower_threshold = np.array([1.0, 1.0, 1.0, 0.2, 0.2, 0.2])
        joint_torque_error_upper_threshold = np.array([1.5, 1.5, 1.5, 0.7, 0.7, 0.7])

        vel_ref_array = np.zeros(6)
        vr = np.zeros(6)
        vt = np.zeros(6)
        pr = np.zeros(6)
        desired_effort = np.zeros(6)
        endeffector_vel = np.zeros(6)
        pose = np.zeros((4,4))
        pose_kdl = np.zeros((4,4))
        pose_rt = np.zeros((3,3))
        wrench = np.zeros(6)
        filtered_wrench = np.zeros(6)
        filtered_wrench_clip = np.zeros(6)
        wrench_global = np.zeros(6)
        filtered_wrench_global = np.zeros(6)
        inertia = np.zeros(6)
        joint_desired_torque = np.zeros(6)
        filtered_joint_desired_torque = np.zeros(6)
        ratio = np.zeros(6)
        rate = rospy.Rate(500)

        self.filter.calculate_initial_values(self.current_wrench)

        start_time = time.time()
        last_time = start_time
        init_pos = deepcopy(self.current_joint_positions)
        last_vel = self.current_joint_velocities

        Ja = self.kdl_kin.jacobian(self.current_joint_positions)
        pose_kdl = self.kdl_kin.forward(self.current_joint_positions)
        pose_rt = pose_kdl[:3,:3]
        wrench = self.current_wrench
        np.matmul(pose_rt, wrench[:3], out = wrench_global[:3])
        np.matmul(pose_rt, wrench[3:], out = wrench_global[3:])
        np.matmul(Ja.transpose(), wrench_global, out = joint_desired_torque)
        recent_data_focus_coeff = 0.99
        p = 1 / recent_data_focus_coeff
        joint_torque_error = joint_desired_torque

        while not self.shutdown and self.safety_mode == 1: #chutdown is set on ctrl-c.

            # jacobian
            Ja = self.kdl_kin.jacobian(self.current_joint_positions)

            pose_kdl = self.kdl_kin.forward(self.current_joint_positions)
            pose_rt = pose_kdl[:3,:3]

            wrench = self.current_wrench
            # filtered_wrench = np.array(self.filter.filter(wrench))
            # np.matmul(pose_rt, filtered_wrench[:3], out = filtered_wrench_global[:3])
            # np.matmul(pose_rt, filtered_wrench[3:], out = filtered_wrench_global[3:])
            # np.matmul(Ja.transpose(), filtered_wrench_global, out = filtered_joint_desired_torque)
            np.matmul(pose_rt, wrench[:3], out = wrench_global[:3])
            np.matmul(pose_rt, wrench[3:], out = wrench_global[3:])
            np.matmul(Ja.transpose(), wrench_global, out = joint_desired_torque)

            if np.any(np.abs(self.current_joint_positions - init_pos)>0.01) or np.any(np.abs(self.current_joint_velocities)>0.001):
                jdt = joint_desired_torque - joint_torque_error
                flag = 1
            else:
                joint_torque_error = joint_torque_error + p / (1 + p) * (joint_desired_torque - joint_torque_error)
                p = (p - p ** 2 / (1 + p)) / recent_data_focus_coeff
                jdt = joint_desired_torque - joint_torque_error
                flag = -1

            # joint inertia
            T1tb = forward_link(np.array([self.current_joint_positions[0], self.DH_d[0], self.DH_a[0], self.DH_alpha[0]]))
            T2t1 = forward_link(np.array([self.current_joint_positions[1], self.DH_d[1], self.DH_a[1], self.DH_alpha[1]]))
            T3t2 = forward_link(np.array([self.current_joint_positions[2], self.DH_d[2], self.DH_a[2], self.DH_alpha[2]]))
            T4t3 = forward_link(np.array([self.current_joint_positions[3], self.DH_d[3], self.DH_a[3], self.DH_alpha[3]]))
            T5t4 = forward_link(np.array([self.current_joint_positions[4], self.DH_d[4], self.DH_a[4], self.DH_alpha[4]]))
            T6t5 = forward_link(np.array([self.current_joint_positions[5], self.DH_d[5], self.DH_a[5], self.DH_alpha[5]]))

            J6 = self.J6l
            J5 = self.J5l + np.matmul(np.matmul(T6t5,J6),T6t5.transpose())
            J4 = self.J4l + np.matmul(np.matmul(T5t4,J5),T5t4.transpose())
            J3 = self.J3l + np.matmul(np.matmul(T4t3,J4),T4t3.transpose())
            J2 = self.J2l + np.matmul(np.matmul(T3t2,J3),T3t2.transpose())
            J1 = self.J1l + np.matmul(np.matmul(T2t1,J2),T2t1.transpose())

            inertia[0] = J1[2,2]
            inertia[1] = J2[2,2]
            inertia[2] = J3[2,2]
            inertia[3] = J4[2,2]
            inertia[4] = J5[2,2]
            inertia[5] = J6[2,2]

            # loop rate check
            current_time = time.time()
            relative_time = current_time - start_time
            loop_time = current_time - last_time
            last_time = current_time
            print(1/loop_time)

            # integration approach
            # joint_torque_error = joint_desired_torque - filtered_joint_desired_torque
            ref_pos = np.sin(1.0 * relative_time) * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.1
            pr = self.current_joint_positions - init_pos
            spring_torque = virtual_stiffness * pr
            # np.clip((np.absolute(joint_torque_error) - joint_torque_error_lower_threshold)/(joint_torque_error_upper_threshold - joint_torque_error_lower_threshold),0,1,ratio)
            # np.clip(np.absolute(spring_torque) * (inertia + self.inertia_offset) / 0.5, 0, 1, ratio) # 1.0 should be larger than the unfiltered joint torque at rest
            # mixed_joint_torque = filtered_joint_desired_torque + joint_torque_error * ratio
            # damper_torque = 2 * zeta * np.sqrt(virtual_stiffness / (inertia + self.inertia_offset)) * pr
            # acc = (mixed_joint_torque - spring_torque - 2 * zeta * np.sqrt(virtual_stiffness * (inertia + self.inertia_offset)) * self.current_joint_velocities) / (inertia + self.inertia_offset)
            acc = (jdt - spring_torque - 2 * zeta * np.sqrt(virtual_stiffness * (inertia + self.inertia_offset)) * self.current_joint_velocities) / (inertia + self.inertia_offset)
            np.clip(acc, -self.max_joint_acc, self.max_joint_acc, acc)
            # if current_time - start_time > 2.0:
            # vr += (acc - spring_torque - virtual_damping * self.current_joint_velocities) / sample_rate
            vr += acc / sample_rate
            np.clip(vr,-self.max_joint_speeds,self.max_joint_speeds,vr)
            # vt = vr - damper_torque
            vt = vr

            # proportional approach
            # pr = self.current_joint_positions - init_pos
            # acc = (self.current_joint_velocities - last_vel) * sample_rate
            # last_vel = self.current_joint_velocities
            # np.clip(acc,-self.max_joint_acc,self.max_joint_acc,acc)
            # if current_time - start_time > 2.0:
                # vr = joint_desired_torque - (inertia + self.inertia_offset) * (acc + virtual_damping * self.current_joint_velocities + virtual_stiffness * pr)
                # np.clip(vr,-self.max_joint_speeds,self.max_joint_speeds,vr)

            # self.test_data.data = jdt
            self.test_data.data = np.array([joint_desired_torque[2], joint_torque_error[2], jdt[2], flag, 0.0, 0.0])
            self.test_data_pub.publish(self.test_data)

            vel_ref_array[2] = vt[2]
            # vel_ref_array[0] = vt[0]
            # vel_ref_array[1] = vt[1]
            # vel_ref_array[3] = vt[3]
            # vel_ref_array[4] = vt[4]
            # vel_ref_array[5] = vt[5]
            #vel_ref_array = vt
            #print(vr[2])
            np.clip(vel_ref_array,-self.max_joint_speeds,self.max_joint_speeds,vel_ref_array)
            #publish
            self.vel_ref.data = vel_ref_array
            # self.ref_vel_pub.publish(self.vel_ref)
            self.vel_pub.publish(self.vel_ref)
            #wait
            rate.sleep()
        self.stop_arm(safe = True)

    def run(self):
        '''Run runs the move routine repeatedly, accounting for the
        enable/disable switch'''

        print('Put the control arm in start configuration.')
        print('Depress and hold the deadman switch when ready to move.')

        while not rospy.is_shutdown():
            #check safety
            if not self.safety_mode == 1:
                time.sleep(0.01)
                continue
            #start moving
            print('Starting Free Movement')
            self.move(capture_start_as_ref_pos = True,
                      dialoge_enabled = False)

if __name__ == "__main__":
    #This script is included for testing purposes
    print("starting")

    arm = ur5e_admittance(conservative_joint_lims = False)
    time.sleep(1)
    arm.stop_arm()

    print(arm.move_to_robost(arm.default_pos,
                             speed = 0.1,
                             override_initial_joint_lims=True,
                             require_enable = True))

    raw_input("Hit enter when ready to move")
    arm.run()

    arm.stop_arm()
