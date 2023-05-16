#!/usr/bin/env python3

import rospy

from visualization_msgs.msg import Marker

def createMarkers(posestampedmsg):
    topic = 'test_marker'
    publisher = rospy.Publisher(topic, Marker,queue_size=2)

    # rospy.init_node('register')

    while not rospy.is_shutdown():

        marker = Marker()
        marker.header.frame_id = posestampedmsg.header.frame_id
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.position.x = posestampedmsg.pose.position.x
        marker.pose.position.y = posestampedmsg.pose.position.y
        marker.pose.position.z = posestampedmsg.pose.position.z
        marker.pose.orientation.x = posestampedmsg.pose.orientation.x
        marker.pose.orientation.y = posestampedmsg.pose.orientation.y
        marker.pose.orientation.z = posestampedmsg.pose.orientation.z
        marker.pose.orientation.w = posestampedmsg.pose.orientation.w

        publisher.publish(marker)


 