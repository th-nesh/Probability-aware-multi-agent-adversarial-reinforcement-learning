#!/usr/bin/env python3
import math
import tf

def polar2Cartesion(r,theta):

    print("Converting to cartesian form")
    radian = math.radians(theta)

    x=r*math.cos(radian)
    y=r*math.sin(radian)

    return x,y

def euler2Quaternions(roll, pitch, yaw):

    print("Converting to euler form")
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)

    #we print these three angles
    #print('roll = ', roll, 'pitch = ', pitch,'yaw = ', yaw)

    quaternion = tf.transformations.quaternion_from_euler(roll_rad, pitch_rad, yaw_rad)

    return quaternion