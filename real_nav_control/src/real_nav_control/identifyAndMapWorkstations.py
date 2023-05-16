#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from pyzbar.pyzbar import decode


from real_nav_control.getDistAngleLidar import getDistAngle
from real_nav_control.conversions import polar2Cartesion
from tf_laserlink2map import laser2mapConv 
from createMarker import createMarkers
import math

import json

#Global variable to read camera data
Image_data = []

#Global counter to save image
image_counter = 0   

#Global datcollection
target_identified = {}

def shutdownProcess():
    global target_identified
    print("_______________________________________End___________________________________________")
    print(target_identified)
    file = "/home/cpfactory/vaibhav_data/catkin_ws/src/IDT/real_nav_control/param/markers.json"
    with open(file, 'w') as jsonfile:
        json.dump(target_identified,jsonfile,sort_keys=True,indent=4)
    
    print("**********************file Saved**************************")


def callback(data):
    global image_counter
    global target_identified
    image_counter += 1

    br = CvBridge()

    rospy.logdebug("Recieving video frames")

    img = br.imgmsg_to_cv2(data, "rgb8")
    
    width = int(data.width * 0.80)
    height = int(data.height * 0.80)
    dim = (width, height)

    img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    for barcode in decode(img_resized):

        center_x_value =0

        for iter in range(1,151):

            mydata = barcode.data.decode('utf-8')
            # print(barcode.data)
            pts = np.array([barcode.polygon],np.int32)
            pts = pts.reshape(-1,1,2)
            cv2.polylines(img_resized,[pts],True,(255,0,255),5)
            pts2 = barcode.rect

            #Calculate the center of the QR code
            center_x = pts2[0]+(pts2[2]/2)
            print(iter,"    " ,center_x)
            center_x_value = center_x + center_x_value
        
        
        #Calculate the angle of the target wrt to camera
        camera_angle =  36 - (0.1125*(center_x_value / 150 ))
        

        #Calculate the distance of the angle wrt the given camera angle
        objectDist = getDistAngle(camera_angle)

        #Find the x and y axis of the target wrt to laser frame
        object_x, object_y = polar2Cartesion(objectDist,camera_angle)

        print(target_identified.keys())

        if mydata in target_identified.keys():
            print(mydata, "is already identified")
        else:
            print("Storing data....")
            # target_identified[mydata] = {"object_x" : object_x, "object_y": object_y, "object_z": 0.0, "camera_angle": camera_angle}
            print(mydata)
            posedstamped = PoseStamped()
            posedstamped = laser2mapConv(object_x, object_y, 0.0, roll = camera_angle, pitch = 0.0, yaw = 0.0)
            target_identified[mydata] = {"posedstamped" : {
                "header" : {
                "seq" : posedstamped.header.seq,
                "stamp" : str(posedstamped.header.stamp),
                "frame_id": posedstamped.header.frame_id,
                    },
                "pose": {
                        "position": {
                            "x": posedstamped.pose.position.x,
                            "y": posedstamped.pose.position.y,
                            "z": posedstamped.pose.position.z,
                            },
                        "orientation": {
                            "x": posedstamped.pose.orientation.x,
                            "y": posedstamped.pose.orientation.y,
                            "z": posedstamped.pose.orientation.z,
                            "w": posedstamped.pose.orientation.w,
                            }
                        } 
                    }
                }
            print(posedstamped)

        cv2.putText(img_resized, mydata,(pts2[0],pts2[1]), cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,255),2)
    
    cv2.imshow('Camera', img_resized)
    cv2.waitKey(1)


    


if __name__ == '__main__':    
    # object_z = 0.0
    try:
        
       # Initializes a rospy node to identify and map workstations on map
        rospy.init_node('identifyAndMapWorkStations', disable_signals= True)
        camera_sub = rospy.Subscriber('image_raw', Image, callback)

        rate = rospy.Rate(500)
        while not rospy.is_shutdown():
            # print("_______________________________________Start___________________________________________")
            if (len(target_identified.keys()) == 4):
                camera_sub.unregister()
                print(target_identified.keys())
                rospy.signal_shutdown("All targets identified")
                rospy.on_shutdown(shutdownProcess)
            rate.sleep()
 
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")
