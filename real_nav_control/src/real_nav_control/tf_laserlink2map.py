#!/usr/bin/env python3
import rospy
import tf
from geometry_msgs.msg import PoseStamped
import conversions

def laser2mapConv(x, y, z, roll, pitch, yaw):

    print("Transforming to map frame")
    #init node
    # rospy.init_node('laser2mapConv')

    #create a new transform listerner
    transform_listener = tf.TransformListener()
    transform_listener.waitForTransform('/laser_link','/map',rospy.Time(),rospy.Duration(4.0))

    time = rospy.get_rostime()
    currentTime = time.secs    
    i=0

    while i<1:

        posedstamped = PoseStamped()
        posedstamped.header.frame_id = 'laser_link'
        posedstamped.header.stamp = rospy.Time(0)
        posedstamped.pose.position.x = x
        posedstamped.pose.position.y = y
        posedstamped.pose.position.z = z

        quanternion = conversions.euler2Quaternions(roll, pitch, yaw)

        posedstamped.pose.orientation.x = quanternion[0]
        posedstamped.pose.orientation.y = quanternion[1]
        posedstamped.pose.orientation.z = quanternion[2]
        posedstamped.pose.orientation.w = quanternion[3]

        try:
            print("Saving data in posedstamped form")
            posedstamped_map = transform_listener.transformPose('map', posedstamped)
            print("posedstamped_map")
            i=i+2
            print(i)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("Exception faced")
            continue

    return posedstamped_map