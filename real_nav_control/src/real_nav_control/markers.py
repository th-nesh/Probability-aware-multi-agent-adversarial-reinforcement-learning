#!/usr/bin/env python3

import rospy

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import json

#Markers position
markersPosition = {}

def createMarkers(posestampedmsg):
    global markersPosition

    topic = 'target_markers'
    publisher = rospy.Publisher(topic, MarkerArray)

    rospy.init_node('PublishTargetMarkers')

    numOfTargets = len(posestampedmsg.keys())
    markerArray = MarkerArray()

    count = 0

    

    while not rospy.is_shutdown():

            for key in markersPosition.keys():
                marker = Marker()
                marker.ns = key
                marker.id = target_id

                target_id= target_id
                marker.header.frame_id = markersPosition[key]["posedstamped"]["header"]["frame_id"]
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.pose.position.x = markersPosition[key]["posedstamped"]["header"]["frame_id"]["pose"]["position"]["x"]
                marker.pose.position.y = markersPosition[key]["posedstamped"]["header"]["frame_id"]["pose"]["position"]["y"]
                marker.pose.position.z = markersPosition[key]["posedstamped"]["header"]["frame_id"]["pose"]["position"]["z"]
                marker.pose.orientation.x = markersPosition[key]["posedstamped"]["header"]["frame_id"]["pose"]["orientation"]["x"]
                marker.pose.orientation.y = markersPosition[key]["posedstamped"]["header"]["frame_id"]["pose"]["orientation"]["y"]
                marker.pose.orientation.z = markersPosition[key]["posedstamped"]["header"]["frame_id"]["pose"]["orientation"]["z"]
                marker.pose.orientation.w = markersPosition[key]["posedstamped"]["header"]["frame_id"]["pose"]["orientation"]["w"]

                markerArray.markers.append(marker)

            target_id = 0
            for marker in markerArray.markers:
                marker.id = target_id
                target_id += 1

            publisher.publish(markerArray)


if __name__ == '__main__':    
    file = "/home/cpfactory/vaibhav_data/catkin_ws/src/IDT/real_nav_control/param/markers.json"
    with open(file, 'r') as jsonfile:
        markersPosition = json.load(jsonfile)

    createMarkers(markersPosition)
    #print(len(markersPosition.keys()))
    