#!/usr/bin/env python3

import rospy
import numpy as np
import time
import math
from sympy import symbols, cos, sin, Matrix, lambdify, diff
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

rospy.init_node("velocity_control_gazebo", anonymous=True)

q1, q2, q3, q4, q5, q6, q7 = symbols("q1 q2 q3 q4 q5 q6 q7")
q = [q1, q2, q3, q4, q5, q6, q7]
T_prev = Matrix.eye(4)
J_v = Matrix.zeros(3, 7)
J_o = Matrix.zeros(3, 7)
rho = 1e-6
identity_matrix = np.eye(6)
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
n = np.zeros(3)

# https://www.kinovarobotics.com/uploads/User-Guide-Gen3-R07.pdf
BT1 = Matrix([
    [cos(q1), -sin(q1),  0,      0],
    [-sin(q1), -cos(q1), 0,      0],
    [0,        0,       -1,   0.1564],
    [0,        0,        0,      1]
])

T12 = Matrix([
    [cos(q2), -sin(q2),  0,      0],
    [0,        0,       -1,   0.0054],
    [sin(q2),  cos(q2),  0,  -0.1284],
    [0,        0,        0,      1]
])

T23 = Matrix([
    [cos(q3), -sin(q3),  0,      0],
    [0,        0,        1,  -0.2104],
    [-sin(q3), -cos(q3), 0,  -0.0064],
    [0,        0,        0,      1]
])

T34 = Matrix([
    [cos(q4), -sin(q4),  0,      0],
    [0,        0,       -1,   0.0064],
    [sin(q4),  cos(q4),  0,  -0.2104],
    [0,        0,        0,      1]
])

T45 = Matrix([
    [cos(q5), -sin(q5),  0,      0],
    [0,        0,        1,  -0.2084],
    [-sin(q5), -cos(q5), 0,  -0.0064],
    [0,        0,        0,      1]
])

T56 = Matrix([
    [cos(q6), -sin(q6),  0,      0],
    [0,        0,       -1,      0],
    [sin(q6),  cos(q6),  0,  -0.1059],
    [0,        0,        0,      1]
])

T67 = Matrix([
    [cos(q7), -sin(q7),  0,      0],
    [0,        0,        1,  -0.1059],
    [-sin(q7), -cos(q7), 0,      0],
    [0,        0,        0,      1]
])

T7P = Matrix([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, -0.0615],
    [0, 0, 0, 1]
])    # T7P is end_effector_joint to adaptor, currently (0, 0, -0.0615) m

T_list = [BT1, T12, T23, T34, T45, T56, T67, T7P]
T_total = BT1 * T12 * T23 * T34 * T45 * T56 * T67 * T7P
p_end = T_total[:3, 3]

for i in range(7):
    T_prev = T_prev * T_list[i]
    z_prev = T_prev[:3, 2]
    p_prev = T_prev[:3, 3]
    J_v[:, i] = z_prev.cross(p_end - p_prev)
    J_o[:, i] = z_prev

J_total = J_v.col_join(J_o)
J_func = lambdify(q, J_total, "numpy")
T_total_func = lambdify(q, T_total, "numpy")

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

def cal_J_pseudo(q_values, rho, identity_matrix):
    J_numeric = J_func(*q_values)
    J_transpose = J_numeric.T
    J_pseudo = J_transpose @ np.linalg.inv(J_numeric @ J_transpose + rho * identity_matrix)
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

def cal_p_dot(target_pose, T_numeric, j_pseudo):
    position_error = target_pose[:3] - T_numeric[:3, 3]
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

def cal_omega(target_pose, T_numeric, j_pseudo):
    R_d = euler_to_rotation_matrix(*target_pose[3:])
    R_e = np.dot(R_d, T_numeric[:3, :3].T)
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
        j_pseudo = cal_J_pseudo(q_values, rho, identity_matrix)
        T_numeric = T_total_func(*q_values)
        p_dot = cal_p_dot(target_pose, T_numeric, j_pseudo)
        omega = cal_omega(target_pose, T_numeric, j_pseudo)
        desired_twist = np.concatenate((p_dot, omega))
        joint_velocities = j_pseudo @ desired_twist
        
        joint_vel_msg = Float64MultiArray()
        joint_vel_msg.data = joint_velocities.tolist()
        joint_vel_pub.publish(joint_vel_msg)

    rate.sleep()
