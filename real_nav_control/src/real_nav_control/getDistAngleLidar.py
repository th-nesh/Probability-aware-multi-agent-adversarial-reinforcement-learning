#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
import math

def getDistAngle(theta):

    print("Fetching Distance .......")
    # rospy.init_node('listener', anonymous=True)
    # objectDist = []
    laserMsg =rospy.wait_for_message('/scan', LaserScan, 2)
    
    # for angle in theta:

    #     objectDist.append(laserMsg.ranges[math.floor((angle * 2)+240)]) 

    objectDist = laserMsg.ranges[math.floor((theta * 2)+240)]

    return objectDist

getDistAngle(0)
