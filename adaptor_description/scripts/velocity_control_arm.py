#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from kortex_driver.msg import Base_JointSpeeds, JointSpeed

rospy.init_node("velocity_control_arm", anonymous=True)

current_j_pseudo = None
current_direct_kinematics = None
target_pose = None
v_max = 0.1
v_min = 0.005
omega_max = 0.08
omega_min = 0.005
Threshold_p = 0.001
Threshold_omega = 0.001
lamda_p = 20
lamda_omega = 20
n = np.zeros(3)

def j_pseudo_callback(data):
    global current_j_pseudo
    current_j_pseudo = np.array(data.data).reshape(7, 6)

def direct_kinematics_callback(data):
    global current_direct_kinematics
    current_direct_kinematics = np.array(data.data).reshape(4, 4)

def target_pose_callback(data):
    global target_pose
    target_pose = np.array(data.data)

rospy.Subscriber("/resolved_rates/j_pseudo", Float64MultiArray, j_pseudo_callback)
rospy.Subscriber("/resolved_rates/direct_kinematics", Float64MultiArray, direct_kinematics_callback)
rospy.Subscriber("/target_pose", Float64MultiArray, target_pose_callback)

joint_vel_pub = rospy.Publisher("/my_gen3/in/joint_velocity", Base_JointSpeeds, queue_size=10)

def euler_to_rotation_matrix(roll, pitch, yaw):
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    R = R_z @ R_y @ R_x
    return R

def cal_p_dot(target_pose, current_direct_kinematics, current_j_pseudo):
    position_error = target_pose[:3] - current_direct_kinematics[:3, 3]
    delta_p = np.linalg.norm(position_error)
    if delta_p < Threshold_p:
        p_dot = np.array([0.0, 0.0, 0.0])
    else:
        n_hat = position_error / delta_p
        if delta_p / Threshold_p > lamda_p:
            v_module = v_max
        else:
            v_module = v_min + (v_max - v_min) * (delta_p - Threshold_p) / (Threshold_p * (lamda_p - 1))
        p_dot = v_module * n_hat
    return p_dot

def cal_omega(target_pose, current_direct_kinematics, current_j_pseudo):
    R_d = euler_to_rotation_matrix(*target_pose[3:])
    R_e = np.dot(R_d, current_direct_kinematics[:3, :3].T)
    theta_e = np.arccos((np.trace(R_e) - 1) / 2)
    if theta_e < Threshold_omega:
        omega_d = np.array([0.0, 0.0, 0.0])
    else:
        if np.sin(theta_e) > 1e-6:
            m_e = np.array([
                (R_e[2, 1] - R_e[1, 2]) / (2 * np.sin(theta_e)),
                (R_e[0, 2] - R_e[2, 0]) / (2 * np.sin(theta_e)),
                (R_e[1, 0] - R_e[0, 1]) / (2 * np.sin(theta_e))
            ])
        else:
            m_e = np.array([0, 0, 0])
        delta_omega = theta_e
        if delta_omega / Threshold_omega > lamda_omega:
            omega_module = omega_max
        else:
            omega_module = omega_min + (omega_max - omega_min) * (delta_omega - Threshold_omega) / (Threshold_omega * (lamda_omega - 1))
        omega_d = omega_module * m_e
    return omega_d

rate = rospy.Rate(40)

while not rospy.is_shutdown():
    if current_j_pseudo is not None and current_direct_kinematics is not None and target_pose is not None:
        
        p_dot = cal_p_dot(target_pose, current_direct_kinematics, current_j_pseudo)
        omega = cal_omega(target_pose, current_direct_kinematics, current_j_pseudo)
        desired_twist = np.concatenate((p_dot, omega))
        
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
