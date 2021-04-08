#! /usr/bin/env python

# Import the module
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics

robot = URDF.from_xml_file("/home/chunpeng/catkin_ws/src/ur5e_admittance_control/config/ur5e.urdf")

tree = kdl_tree_from_urdf_model(robot)
print tree.getNrOfSegments()

chain = tree.getChain("base_link", "ee_link")
print chain.getNrOfJoints()

# forwawrd kinematics
kdl_kin = KDLKinematics(robot, "base_link", "ee_link")
q = [0, 0, 1, 0, 1, 0]

pose = kdl_kin.forward(q) # forward kinematics (returns homogeneous 4x4 matrix)
print pose

q_ik = kdl_kin.inverse(pose) # inverse kinematics
print "iverse\n",q_ik

if q_ik is not None:
    pose_sol = kdl_kin.forward(q_ik) # should equal pose
    print pose_sol

J = kdl_kin.jacobian(q)
print 'J:', J
