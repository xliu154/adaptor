#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from kortex_driver.msg import Base_JointSpeeds, JointSpeed

rospy.init_node("vel_control", anonymous=True)

current_j_pseudo = None
current_position = None
current_target_pose = None

def j_pseudo_callback(data):
    global current_j_pseudo
    current_j_pseudo = np.array(data.data).reshape(7, 6)

def position_callback(data):
    global current_position
    current_position = np.array(data.data)

def target_pose_callback(data):
    global current_target_pose
    current_target_pose = np.array(data.data)

rospy.Subscriber("/resolved_rates/j_pseudo", Float64MultiArray, j_pseudo_callback)
rospy.Subscriber("/resolved_rates/position", Float64MultiArray, position_callback)
rospy.Subscriber("/target_pose", Float64MultiArray, target_pose_callback)

joint_vel_pub = rospy.Publisher("/my_gen3/in/joint_velocity", Base_JointSpeeds, queue_size=10)

rate = rospy.Rate(40)

kv = 0.05
kxi = 0.05

while not rospy.is_shutdown():
    if current_j_pseudo is not None and current_position is not None and current_target_pose is not None:
        position_error = current_target_pose[:3] - current_position[:3]
        orientation_error = current_target_pose[3:] - current_position[3:]
        linear_velocity = kv * position_error
        angular_velocity = kxi * orientation_error
        desired_twist = np.concatenate((linear_velocity, angular_velocity))
        joint_velocities = current_j_pseudo @ desired_twist
        joint_velocities = joint_velocities.flatten()
        
        joint_speeds_msg = Base_JointSpeeds()
        for joint_id in range(7):
            joint_speed = JointSpeed()
            joint_speed.joint_identifier = joint_id
            joint_speed.value = joint_velocities[joint_id]
            joint_speed.duration = 1
            
            joint_speeds_msg.joint_speeds.append(joint_speed)
            
        joint_vel_pub.publish(joint_speeds_msg)
    rate.sleep()
