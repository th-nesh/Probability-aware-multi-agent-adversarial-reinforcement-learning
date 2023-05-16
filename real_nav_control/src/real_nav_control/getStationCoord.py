#!/usr/bin/env python3

from __future__ import print_function

from visualization_msgs.msg import Marker
import rospy

def getStation1Coordinate():

    print ("Station 1 Cordinate")
    return rospy.wait_for_message('/station1_marker', Marker, 2)

def getStation2Coordinate():

    print ("Station 2 Cordinate")
    return rospy.wait_for_message('/station2_marker', Marker, 2)

def getStation3Coordinate():

    print ("Station 3 Cordinate")
    return rospy.wait_for_message('/station3_marker', Marker, 2)


def station_coord():

    rospy.init_node('getstationCoordinate', anonymous=True)
    print("Station Coordinates")
    rospy.spin()
    return 



if __name__ == "__main__":
    station_coord()
