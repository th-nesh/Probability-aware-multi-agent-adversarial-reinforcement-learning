#!/usr/bin/env python3

from __future__ import print_function

from real_nav_control.srv import stationCoord, stationCoordResponse
from getStationCoord import getStation1Coordinate, getStation2Coordinate, getStation3Coordinate

import rospy


def getCoord(req):

    print("getCoord")

    if req.station == 1:
        print("1")
        msg = getStation1Coordinate()
    elif req.station == 2:
        msg = getStation2Coordinate()
    elif req.station== 3:
        msg = getStation3Coordinate()

    return msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, 0

def stationCoordinate_server():
    rospy.init_node('stationCoord_server')
    s = rospy.Service('stationCoordinates', stationCoord, getCoord)

    print("Ready to Station Coordinates")
    rospy.spin()

if __name__ == "__main__":
    stationCoordinate_server()
