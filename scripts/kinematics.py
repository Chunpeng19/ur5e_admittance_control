#! /usr/bin/env python
import rospy
import numpy as np
import time
from copy import deepcopy

from ur_kinematics.ur_kin_py import forward, inverse

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Header


joint_inversion = np.array([-1,-1,1,-1,-1,-1])
joint_reorder = [2,1,0,3,4,5]
current_joint_positions = np.zeros(6)
current_joint_velocities = np.zeros(6)
lower_lims = (np.pi/180)*np.array([0.0, -100.0, 0.0, -180.0, -180.0, 90.0])
upper_lims = (np.pi/180)*np.array([180.0, 0.0, 175.0, 0.0, 0.0, 270.0])

def analytical_ik(pose, upper_lims, lower_lims):
    '''Light wrapper around the ur_kin_py ik call, to return the full set of
    8 ik solutions, modified from the raw output to fall within the upper and
    lower joint lims if possible. If a configuration exceeds joint lims, it is not returned'''

    solns = inverse(pose,0.0)
    # print('Raw Solutions')
    # print(solns)
    # Change joint positions such that they are within the limits.
    larger2Pi = np.logical_and(solns > upper_lims, solns - 2 * np.pi > lower_lims)
    solns[np.where(larger2Pi)] -= 2 * np.pi

    smaller2Pi = np.logical_and(solns < lower_lims, solns + 2 * np.pi < upper_lims)
    solns[np.where(smaller2Pi)] += 2 * np.pi

    # Check if the config is within the joint limits.
    inLimits = np.logical_and(solns >= lower_lims, solns <= upper_lims)
    # print('in limmits')
    # print(inLimits)
    good_solns = solns[np.all(inLimits,1)]
    # print(isInLimits)
    return good_solns

def nearest_ik_solution(solutions, current_joints, threshold = None):
    '''Out of a set of joint solutions, returns the set closest to current_joints.
    If only one solution is input, it is directly output. When the threshold is set,
    only a solution in which the difference between all joints is below the threshold
    may be returned. If no passing solution is found, then None is returned'''

    #ensure compatible shape
    try:
        solutions = solutions.reshape(-1,6)
    except ValueError:
        print('Solutions have bad shape {}'.format(solutions.shape))
        return None

    #remove NaN solutions
    solutions = solutions[np.logical_not(np.any(np.isnan(solutions),1))]

    #check if there are any solutions
    if solutions.size == 0:
        print('All solutions have NaN')
        return None

    #if there are still good solutions, select the best one
    error = np.abs(solutions - current_joints)
    if not threshold is None: #check that the max error in each case is below the thresh
        max_error_per_soln = np.max(error,1)
        solutions = solutions[max_error_per_soln < threshold] #remove extra solutions
        if solutions.size == 0:
            print('Error threshold not met for error {}'.format(error))
            return None
    error_sum_per_soln = np.sum(error,1)
    min_error_soln = solutions[np.argmin(error_sum_per_soln)]

    return min_error_soln.reshape(-1)

def main():
    rospy.init_node('test_kin', anonymous=True)
    rospy.Subscriber("joint_states", JointState, joint_state_callback)
    time.sleep(1)

    print('Joints')
    print(current_joint_positions)
    # current_pose = forward(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    current_pose = forward(current_joint_positions)
    print('Pose:')
    print(current_pose)
    pose = np.eye(4)
    pose[:3,3] = current_pose[:3,3] + [0, 0, 0.1]
    # print('commanded pose:')
    # print(pose)
    print('solutions')
    # sol = inverse(current_pose,0.0)
    # print(sol)
    good_solutions = analytical_ik(current_pose,upper_lims,lower_lims)
    print(good_solutions)

    print(nearest_ik_solution(np.array([good_solutions,good_solutions]), current_joint_positions, threshold = None))

    # # Change joint positions such that they are within the limits.
    # larger2Pi = np.logical_and(sol > upper_lims, sol - 2 * np.pi > lower_lims)
    # sol[np.where(larger2Pi)] -= 2 * np.pi
    #
    # smaller2Pi = np.logical_and(sol < lower_lims, sol + 2 * np.pi < upper_lims)
    # sol[np.where(smaller2Pi)] += 2 * np.pi
    #
    # # Check if the config is within the joint limits.
    # inLimits = np.logical_and(sol >= lower_lims, sol <= upper_lims)
    # isInLimits = (sum(inLimits) == sol.shape[0])
    # print(isInLimits)
    # print(sol)


    # while not rospy.is_shutdown():
    #     time.sleep(1)
    #     pose = forward(current_joint_positions)
    #     print(pose[:3,3])

def joint_state_callback(data):
    current_joint_positions[joint_reorder] = data.position
    current_joint_velocities[joint_reorder] = data.velocity
    # print(current_joint_positions)


if __name__=="__main__":
    print('running')
    main()
