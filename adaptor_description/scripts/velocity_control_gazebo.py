#!/usr/bin/env python3

import rospy
import time
import math
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

rospy.init_node("velocity_control_gazebo", anonymous=True)

current_joint_positions = None
target_pose = None

v_max = 0.1
v_min = 0.005
omega_max = 0.1
omega_min = 0.005
Threshold_p = 0.001
Threshold_omega = 0.001
lamda_p = 20
lamda_omega = 20

def joint_states_callback(data):
    global current_joint_positions
    current_joint_positions = np.array(data.position)

def target_pose_callback(data):
    global target_pose
    target_pose = np.array(data.data)

rospy.Subscriber("/joint_states", JointState, joint_states_callback)
rospy.Subscriber("/target_pose", Float64MultiArray, target_pose_callback)
joint_vel_pub = rospy.Publisher("/gen3_adaptor_controller/command", Float64MultiArray, queue_size=10)
rate = rospy.Rate(40)

# https://www.kinovarobotics.com/uploads/User-Guide-Gen3-R07.pdf
def cal_T_list(q_values):
    q1, q2, q3, q4, q5, q6, q7 = q_values

    BT1 = np.array([
        [np.cos(q1), -np.sin(q1),  0,      0],
        [-np.sin(q1), -np.cos(q1), 0,      0],
        [0,           0,          -1,   0.1564],
        [0,           0,           0,      1]
    ])

    T12 = np.array([
        [np.cos(q2), -np.sin(q2),  0,      0],
        [0,           0,          -1,   0.0054],
        [np.sin(q2),  np.cos(q2),  0,  -0.1284],
        [0,           0,           0,      1]
    ])

    T23 = np.array([
        [np.cos(q3), -np.sin(q3),  0,      0],
        [0,           0,           1,  -0.2104],
        [-np.sin(q3), -np.cos(q3), 0,  -0.0064],
        [0,           0,           0,      1]
    ])

    T34 = np.array([
        [np.cos(q4), -np.sin(q4),  0,      0],
        [0,           0,          -1,   0.0064],
        [np.sin(q4),  np.cos(q4),  0,  -0.2104],
        [0,           0,           0,      1]
    ])

    T45 = np.array([
        [np.cos(q5), -np.sin(q5),  0,      0],
        [0,           0,           1,  -0.2084],
        [-np.sin(q5), -np.cos(q5), 0,  -0.0064],
        [0,           0,           0,      1]
    ])

    T56 = np.array([
        [np.cos(q6), -np.sin(q6),  0,      0],
        [0,           0,          -1,      0],
        [np.sin(q6),  np.cos(q6),  0,  -0.1059],
        [0,           0,           0,      1]
    ])

    T67 = np.array([
        [np.cos(q7), -np.sin(q7),  0,      0],
        [0,           0,           1,  -0.1059],
        [-np.sin(q7), -np.cos(q7), 0,      0],
        [0,           0,           0,      1]
    ])

    T7P = np.array([
        [1,  0,  0,      0],
        [0, -1,  0,      0],
        [0,  0, -1, -0.0615],
        [0,  0,  0,      1]
    ])

    return [BT1, T12, T23, T34, T45, T56, T67, T7P]

def cal_T_total(q_values, T_list):
    T_total = np.eye(4)
    for T in T_list:
        T_total = np.dot(T_total, T)
    return T_total

def cal_J_pseudo(q_values, T_list, T_total):
    J_v = np.zeros((3, 7))
    J_o = np.zeros((3, 7))
    T_prev = np.eye(4)
    for i in range(7):
        T_prev = np.dot(T_prev, T_list[i])
        z_prev = T_prev[:3, 2]
        p_prev = T_prev[:3, 3]
        J_v[:, i] = np.cross(z_prev, T_total[:3, 3] - p_prev)
        J_o[:, i] = z_prev
    J_total = np.vstack((J_v, J_o))
    J_transpose = J_total.T
    J_pseudo = J_transpose @ np.linalg.inv(J_total @ J_transpose + 1e-6 * np.eye(6))
    return J_pseudo

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

def cal_p_dot(target_pose, T_total):
    position_error = target_pose[:3] - T_total[:3, 3]
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

def cal_omega(target_pose, T_total):
    R_d = euler_to_rotation_matrix(*target_pose[3:])
    R_e = np.dot(R_d, T_total[:3, :3].T)
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

while not rospy.is_shutdown():
    if current_joint_positions is not None and target_pose is not None:
        q_values = current_joint_positions
        T_list = cal_T_list(q_values)
        T_total = cal_T_total(q_values,T_list)
        j_pseudo = cal_J_pseudo(q_values, T_list, T_total)
        p_dot = cal_p_dot(target_pose, T_total)
        omega = cal_omega(target_pose, T_total)
        joint_velocities = j_pseudo @ np.concatenate((p_dot, omega))
        
        joint_vel_msg = Float64MultiArray()
        joint_vel_msg.data = joint_velocities.tolist()
        joint_vel_pub.publish(joint_vel_msg)

    rate.sleep()
