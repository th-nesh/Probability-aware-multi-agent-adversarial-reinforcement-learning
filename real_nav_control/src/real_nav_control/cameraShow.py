#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

#Global variable to read camera data
Image_data = []

#Global counter to save image
image_counter = 0   

filename = 'video_t.avi'
frames_per_seconds = 24.0
res = '480p'

def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

STD_DIMENSIONS = {
    "480p" : (640, 480),
    "720p" : (1280, 720),
   "1080p" : (1920, 1080),

 }


def get_dims(caps, res='1080p'):
     width, height = STD_DIMENSIONS["480p"]
     if res in STD_DIMENSIONS:
         width, height = STD_DIMENSIONS[res]

     change_res(caps, width, height)

     return width, height

#---------------------------------------------------Camera Callback-----------------------------------------------
def Image_Callback(img_data):
    global Image_data
    global image_counter
    image_counter += 1
    bridge = CvBridge()

    cv_image = bridge.imgmsg_to_cv2(img_data,"rgb8")
    
    #Resize Image to 640 * 480 - anything smaller than that wont work with DenseDepth Model
    width = int(img_data.width * 0.80)
    height = int(img_data.height * 0.80)
    dim = (width, height)
    img_resized = cv2.resize(cv_image, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow('Camera', cv_image)
    if image_counter %20 == 0:
        print(image_counter)
        Save_Camera_Input = '/home/cpfactory/backup/thinesh_ws/src/IDT/real_nav_control/src/real_nav_control/Camera_Data/Data' + str(image_counter) + '.png'
        cv2.imwrite(Save_Camera_Input, img_resized)


    cv2.waitKey(1)

    Image_data = ['Image_Data',
                 dim, # dimensions of resized image
                 img_resized,  # image data
                 img_data.header.stamp]

def obstacle_detection():
    global Image_data
    global image_counter

    #--------------------- Camera Obstacle Detection ------------------------------------
    # Input images
    image_counter += 1

    Save_Camera_Input = 'Camera_data/Camera' + str(image_counter) + '.png'

    rospy.sleep(10)

    cv2.imwrite(Save_Camera_Input, Image_data[2])


def callback(data):
    global image_counter
    image_counter += 1

    br = CvBridge()

    rospy.logdebug("Recieving video frames")

    current_frame = br.imgmsg_to_cv2(data, "rgb8")
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    out = cv2.VideoWriter(filename, fourcc, 25, (640,480))
    out.write(current_frame)
    out.release()

    cv2.imshow('Camera', current_frame)

    if image_counter %20 == 0:
        print(image_counter)
        Save_Camera_Input = '/home/cpfactory/thinesh_ws/src/IDT/Datset/Data' + str(image_counter) + '.png'
        cv2.imwrite(Save_Camera_Input, current_frame)


    cv2.waitKey(1)



def recieve_message():
    rospy.init_node('videoRecorder', anonymous=True)

    rospy.Subscriber('image_raw', Image, Image_Callback)
  
    # cap = cv2.VideoCapture(0)

    # fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    # out = cv2.VideoWriter(filename, fourcc, 25, (640,480))
    # # rospy.sleep(1)
    
    # while(cap.isOpened()):
    # #     obstacle_detection()
    #     ret,frame = cap.read()
    #     if ret:
    #         out.write(frame)
    #         cv2.imshow('Camera', frame)
    #     else:
    #         break
    # cap.release()
    # out.release()

    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == '__main__' :
    recieve_message()