#!/usr/bin/env python3

import rospy
import numpy as np
import time
from sympy import symbols, cos, sin, Matrix, lambdify, diff
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R
import math
from tf.transformations import quaternion_from_matrix

rospy.init_node("jacobian_pinv_arm", anonymous=True)

# Transformation matrix and Jacobian matrix symbolic
# define joint states and direct kinematics chain
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
])    # T7P is end_effector_joint to gripper, currently (0, 0, -0.0615) m

# Initialize parameters
J_v = Matrix.zeros(3, 7) # velocity Jacobian and angle velocity Jacobian 
J_o = Matrix.zeros(3, 7)
T_prev = Matrix.eye(4) # previous T
T_list = [BT1, T12, T23, T34, T45, T56, T67, T7P] # list all T
T_total = BT1 * T12 * T23 * T34 * T45 * T56 * T67 * T7P # total transformation
p_end = T_total[:3, 3] # end effector position

for i in range(7):
    T_prev = T_prev * T_list[i] # update current T
    z_prev = T_prev[:3, 2] # update z axis 
    p_prev = T_prev[:3, 3] # update frame position
    J_v[:, i] = z_prev.cross(p_end - p_prev) # The cross product of the joint axis direction z_prev and the difference between the end and joint positions
    J_o[:, i] = z_prev # rotary joints therefore current z axis direction z_prev

J_total = J_v.col_join(J_o)

# combine, J_total = [J_v; J_o]
J_func = lambdify(q, J_total, "numpy")
T_total_func = lambdify(q, T_total, "numpy")

# initialize pseudo inverse parameters
rho = 1e-6
current_joint_positions = None

def joint_states_callback(data):
    global current_joint_positions
    current_joint_positions = np.array(data.position)

rospy.Subscriber("my_gen3/joint_states", JointState, joint_states_callback)

# this script will publish robot current end effector jacobian pesudo inverse and position
j_pseudo_pub = rospy.Publisher("/resolved_rates/j_pseudo", Float64MultiArray, queue_size=10)
position_pub = rospy.Publisher("/resolved_rates/direct_kinematics", Float64MultiArray, queue_size=10)

rate = rospy.Rate(40) # publish frequency
last_log_time = time.time()

# Main Loop
while not rospy.is_shutdown():
    if current_joint_positions is not None:
        q_values = current_joint_positions
        J_numeric = J_func(*q_values)
        J_transpose = J_numeric.T
        identity_matrix = np.eye(J_numeric.shape[0])
        J_pseudo = J_transpose @ np.linalg.inv(J_numeric @ J_transpose + rho * identity_matrix)
        
        T_numeric = T_total_func(*q_values)
        
        j_pseudo_msg = Float64MultiArray()
        j_pseudo_msg.data = J_pseudo.flatten().tolist()
        j_pseudo_pub.publish(j_pseudo_msg)
        
        position_msg = Float64MultiArray()
        position_msg.data = T_numeric.flatten().tolist()
        position_pub.publish(position_msg)

    rate.sleep()
