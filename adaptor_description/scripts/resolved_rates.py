#!/usr/bin/env python3

import rospy
import numpy as np
import time
from sympy import symbols, cos, sin, Matrix, lambdify, diff
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

rospy.init_node("cal_ht", anonymous=True)

q1, q2, q3, q4, q5, q6, q7 = symbols("q1 q2 q3 q4 q5 q6 q7")
q = [q1, q2, q3, q4, q5, q6, q7]

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
])

T_total = BT1 * T12 * T23 * T34 * T45 * T56 * T67 * T7P

p = T_total[:3, 3]
R = T_total[:3, :3]

J_v = Matrix.zeros(3, 7)
J_omega = Matrix.zeros(3, 7)

for i in range(7):
    J_v[:, i] = diff(p, q[i])

for i in range(7):
    J_omega[:, i] = diff(R[:, 2], q[i])

J_total = J_v.col_join(J_omega)

J_func = lambdify(q, J_total, "numpy")
T_total_func = lambdify(q, T_total, "numpy")

rho = 1e-4

current_joint_positions = None

def joint_states_callback(data):
    global current_joint_positions
    current_joint_positions = np.array(data.position)

rospy.Subscriber("/joint_states", JointState, joint_states_callback)

j_pseudo_pub = rospy.Publisher("/resolved_rates/j_pseudo", Float64MultiArray, queue_size=10)
position_pub = rospy.Publisher("/resolved_rates/position", Float64MultiArray, queue_size=10)
rate = rospy.Rate(24)

last_log_time = time.time()

while not rospy.is_shutdown():
    if current_joint_positions is not None:
        q_values = current_joint_positions

        J_numeric = J_func(*q_values)
        T_numeric = T_total_func(*q_values)
        J_transpose = J_numeric.T
        identity_matrix = np.eye(J_numeric.shape[0])
        J_pseudo = J_transpose @ np.linalg.inv(J_numeric @ J_transpose + rho * identity_matrix)

        position = np.zeros(6)
        position[:3] = T_numeric[:3, 3].flatten()
        position[3:] = [
            np.arctan2(T_numeric[2, 1], T_numeric[2, 2]),
            np.arctan2(-T_numeric[2, 0], np.sqrt(T_numeric[2, 1]**2 + T_numeric[2, 2]**2)),
            np.arctan2(T_numeric[1, 0], T_numeric[0, 0])
        ]
        
        j_pseudo_msg = Float64MultiArray()
        j_pseudo_msg.data = J_pseudo.flatten().tolist()
        j_pseudo_pub.publish(j_pseudo_msg)
        
        position_msg = Float64MultiArray()
        position_msg.data = position.tolist()
        position_pub.publish(position_msg)        
        
        if time.time() - last_log_time >= 15:
            rospy.loginfo("Published Jacobian Matrix Pseudoinverse.")
            last_log_time = time.time()

    rate.sleep()
