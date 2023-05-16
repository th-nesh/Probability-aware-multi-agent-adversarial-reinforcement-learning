#!/usr/bin/env python3

import numpy as np
import time
from PIL import Image, ImageDraw
import sys
import pickle
import torch as T

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist

from transform_image_map import to_pixels, to_ros_coordinates
from Env_Adv1 import Env_Adv1
from ppo import Agent

sys.setrecursionlimit(4000)

                                                                        # 0 0 -> vanilla
adv_active = bool(int(sys.argv[1])) and not bool(int(sys.argv[2]))      # 1 0 -> adversary
adv_active_2 = not bool(int(sys.argv[1])) and bool(int(sys.argv[2]))    # 0 1 -> prot
adv_active_3 = bool(int(sys.argv[1])) and bool(int(sys.argv[2]))        # 1 1 -> prot with adversary

real_data =[]
velocity_publisher_robot = rospy.Publisher('cmd_vel_real', Twist, queue_size=10)



def velCallback(velTwist):
    pass


def realCallback(data):
    global real_data
    real_data = ['real_data',
                 data.pose.pose.position.x, 
                 data.pose.pose.position.y,
                 data.pose.pose.orientation.z,  # this will be a value e[-1, 1] and can be converted [-pi, pi] with angle=arcsin(z)*2
                 data.header.stamp]
    # print(real_data[3])


def move(dist):

    speed = 0.15
    
    msg_test_forward = Twist()
    msg_test_forward.linear.x = speed
    msg_test_forward.linear.y = 0
    msg_test_forward.linear.z = 0
    msg_test_forward.angular.x = 0
    msg_test_forward.angular.y = 0
    msg_test_forward.angular.z = 0

    msg_test_stop = Twist()
    msg_test_stop.linear.x = 0
    msg_test_stop.linear.y = 0
    msg_test_stop.linear.z = 0
    msg_test_stop.angular.x = 0
    msg_test_stop.angular.y = 0
    msg_test_stop.angular.z = 0

    t0 = rospy.Time.now().to_sec()
    current_dist = 0
    dist = np.abs(dist)

    while(current_dist < dist):
        velocity_publisher_robot.publish(msg_test_forward)
        t1 = rospy.Time.now().to_sec()
        current_dist = speed * (t1-t0)
    velocity_publisher_robot.publish(msg_test_stop)


def rotate(angle):  # in this function I work with angle e[0, 2pi] !!!

    rot_speed = 10/(360)*(2*np.pi) # 10 degrees/sec ???

    msg_test_stop = Twist()
    msg_test_stop.linear.x = 0
    msg_test_stop.linear.y = 0
    msg_test_stop.linear.z = 0
    msg_test_stop.angular.x = 0
    msg_test_stop.angular.y = 0
    msg_test_stop.angular.z = 0

    msg_test_rotate = Twist()
    msg_test_rotate.linear.x = 0
    msg_test_rotate.linear.y = 0
    msg_test_rotate.linear.z = 0
    msg_test_rotate.angular.x = 0
    msg_test_rotate.angular.y = 0
    if angle < 0:
        msg_test_rotate.angular.z = -rot_speed
    else:
        msg_test_rotate.angular.z = rot_speed

    t0 = rospy.Time.now().to_sec()
    current_angle = 0
    angle = np.abs(angle)

    while(current_angle < angle):
        velocity_publisher_robot.publish(msg_test_rotate)
        t1 = rospy.Time.now().to_sec()
        current_angle = rot_speed * (t1-t0)
    velocity_publisher_robot.publish(msg_test_stop)


# XXX todo: this kind of seems to work... maybe better check how (not) precise is the rotate() (and move()) function because it might be the problem(?)   
def calc_command(position, current_angle, target):
    current_angle = np.arcsin(current_angle)*2  # from [-1, 1] to [-pi, pi]
    if current_angle < 0:                       # from [-pi, pi] to [0, 2pi]
        current_angle = current_angle + (2*np.pi)
    
    distance = np.linalg.norm(target - position)
    v = 0.1   # the factor f = (distance/cost) can be used to scale the intended speed
    # comm_angle is the rotation angle -> positive angles represent counter clockwise rotation
    beta = np.arccos(np.abs(target[0] - position[0]) / distance)
    print('beta:', beta)
    # important: in comparison to the "coordinates" of the image, the y-Axis is inverted, so it's different from my test-script
    if current_angle > np.pi:   # XXX NOT SURE ABOUT THIS YET!!!
        print('current_angle in rad:', current_angle)
        current_angle = current_angle - 2*np.pi
    if target[0] - position[0] >= 0 and target[1] - position[1] < 0:   # 4. Quadrant
        comm_angle = 2 * np.pi - (beta + current_angle)
        print('4. Quadrant')
    elif target[0] - position[0] >= 0 and target[1] - position[1] >= 0:  # 1. Quadrant
        comm_angle = beta - current_angle
        print('1. Quadrant')
    elif target[0] - position[0] < 0 and target[1] - position[1] < 0:  # 3. Quadrant
        comm_angle = np.pi + beta - current_angle
        print('3. Quadrant')
    else:                                                               # 2. Quadrant
        comm_angle = np.pi - (beta + current_angle)
        print('2. Quadrant')
    if comm_angle > np.pi:
        print('comm_angle was bigger than pi:', comm_angle)
        comm_angle = comm_angle - 2*np.pi

    t = distance/v

    return comm_angle, distance


def navigate_to_point(target):
    global real_data

    position = np.array([real_data[1], real_data[2]])
    current_angle = real_data[3]
    print('position', position)
    print('curr_angle', current_angle)
    comm_angle, comm_distance = calc_command(position, current_angle, target)
    print('comm_angle', comm_angle)
    print('comm_dist', comm_distance)
    print('----')
    print('now moving!!')
    rotate(comm_angle)
    move(comm_distance)
    print('arrived! --- target:', target)
    print('actual position:', real_data)


def main():

    rospy.init_node('test_yannik', anonymous=True)
    rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, realCallback, queue_size=10)
    rospy.Subscriber('/cmd_vel_real', Twist, velCallback, queue_size=10)

    rospy.sleep(1)

    trajectory_vanilla = [(62, 74), (67, 81), (73,99), (87,110), (104,118), (109,125)]
    trajectory_adv = [(62,74), (68,80), (76,98)]
    trajectory_prot = [(62,74), (69, 75), (96, 73), (104, 75), (117, 83), (114, 104), (109, 125)]
    traj_safe_prot_adv = [(62, 74), (69, 76), (96,78), (104,76), (116,85), (111,103), (113,125)]
    # trajectory_vanilla = [(-3, 1)]

    if adv_active:
        trajectory_vanilla = trajectory_adv
        print('mode:', 'adversary')   
    elif adv_active_2:
        trajectory_vanilla = trajectory_prot
        print('mode:', 'prot only')
    elif adv_active_3:
        trajectory_vanilla = traj_safe_prot_adv
        print('mode:', 'prot with adversary')
    else:
        print('mode:', 'vanilla')
    

    # --- follow waypoints ---
    for trajectory_node in trajectory_vanilla:
        # navigate_to_point(trajectory_node)
        #print('navigating to:', to_ros_coordinates(trajectory_node.coordinates[0], trajectory_node.coordinates[1]))
        navigate_to_point(to_ros_coordinates(trajectory_node[0], trajectory_node[1]))
        # navigate_to_point((trajectory_node[0], trajectory_node[1]))


    # rospy.on_shutdown(saveData)
    
    rospy.spin()    # keeps the script "alive", e.g. for reading subscribed topics
    

if __name__ == '__main__':
    main()
