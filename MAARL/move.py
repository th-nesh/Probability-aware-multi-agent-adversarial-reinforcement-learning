


import numpy as np
import time
#from PIL import Image, ImageDraw
import sys
import pickle
import torch as T
from cv_bridge import CvBridge
import cv2
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import torch
from sensor_msgs.msg import Image
real_data =[]
image_counter = 0
velocity_publisher_robot = rospy.Publisher('cmd_vel_real', Twist, queue_size=10)
deviation =0
center = 0
model = torch.hub.load('/home/varthini/catkin_ws/src/IDT/yolov5/yolov5','custom', path='/home/varthini/catkin_ws/src/IDT/yolov5/yolov5/best.pt', force_reload=True, source= "local")
class example:
    staticVariable = True # Access through class

real_data =[]
def realCallback(data):
  
    global real_data
    real_data = ['real_data',
                 data.pose.pose.position.x, 
                 data.pose.pose.position.y,
                 data.pose.pose.orientation.z,  # this will be a value e[-1, 1] and can be converted [-pi, pi] with angle=arcsin(z)*2
                 data.header.stamp]
    
    print("--------------------------------------------------------------------")
    print(real_data[1], real_data[2])

def calc_command(position, current_angle, target):
    current_angle = np.arcsin(current_angle)*2  # from [-1, 1] to [-pi, pi]
    if current_angle < 0:                       # from [-pi, pi] to [0, 2pi]
        current_angle = current_angle + (2*np.pi)
    
    distance = np.linalg.norm(target - position)
    v = 0.1   # the factor f = (distance/cost) can be used to scale the intended speed
    # comm_angle is the rotation angle -> positive angles represent counter clockwise rotation
    #import pdb; pdb.set_trace()
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

def Image_Callback(img_data):
    now = rospy.Time.now()
    if(example.staticVariable):
        global Image_data
        global image_counter
        global model
        #global deviation
        global center
        
        image_counter += 1
        
        bridge = CvBridge()

        cv_image = bridge.imgmsg_to_cv2(img_data,"rgb8")
        
        #Resize Image to 640 * 480 - YOLO was trained in this size
        width = int(img_data.width * 0.80)
        height = int(img_data.height * 0.80)
        dim = (width, height)
        img_resized = cv2.resize(cv_image, dim, interpolation = cv2.INTER_AREA)

        
        results =model(img_resized) #model is applied to the camera images
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        out = plot_boxes(results,img_resized) #plot the bounding boxes to the images
        print(center)
        ''''dev = int((center-0.197)*640)
        if dev >-2 and dev ==0 and dev <2 :
            deviation =0
        elif dev >2 & dev<4:
            deviation  = 4
        elif dev >4:
            deviation = 8
        elif dev<-2 & dev > -4:
            deviation = -4
        elif dev< -4:
            deviation = -8'''''
        
        cv2.imshow('Camera', out) 
        
        #if image_counter %5 == 0:
            #print(image_counter)
            #Save_Camera_Input = '/home/varthini/catkin_ws/src/IDT/Camera/Images/Test15bck' + str(image_counter) + '.png'
            #cv2.imwrite(Save_Camera_Input, img_resized)


        cv2.waitKey(1)

        Image_data = ['Image_Data',
                    dim, # dimensions of resized image
                    img_resized,  # image data
                    img_data.header.stamp]

def plot_boxes(results, frame):
        global center
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        n = len(labels) #no.of classes"
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            
            if labels[i] ==3 :
                print(class_to_label(labels[i]))
                print(labels[i])
                center = (row[0] + row [2])/2
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

def class_to_label(x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        if x ==0:
            y = "workstation"
        elif x == 1:
            y = "storage"
        elif x == 2:
            y = "table"
        elif x == 3:
            y = "box"
        elif x == 4:
            y = "robot"
        elif x == 5:
            y = "chair"
        return y


def main():

    rospy.init_node('test_thinesh', anonymous=True)
    rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, realCallback, queue_size=10)
    rospy.sleep(1)
    #deviation =0
    trajectory_0= [(-0.7153,2.048), (-1.402,2.391), (-2.062,2.018)]
    trajectory_1= [ (-0.9501,0.969),(-1.645,1.44), (-2.062,2.018)]
    trajectory_2= [ (-0.885,1.196), (-2.062,2.018)]
    trajectory_3= [(-0.7153,2.048), (-1.38,2.196),(-2.062,2.018)]
    trajectory_4= [ (-2.062,2.018)]
    print("in camera")
            #now = rospy.Time.now()
    rate = rospy.Rate(0.5)
    sub = rospy.Subscriber('/image_raw', Image, Image_Callback, queue_size=10)
    rate.sleep()
    sub.unregister()
    deviation = -4
    if deviation == 0:
                trajectory = trajectory_0
                print("trajectory_0")
    elif deviation == 4:
                trajectory = trajectory_1
                print("trajectory_1")
    elif deviation == 8:
                trajectory = trajectory_2
                print("trajectory_2")
    elif deviation == -4:
                trajectory = trajectory_3
                print("trajectory_3")
    elif deviation == -8:
                trajectory = trajectory_4
                print("trajectory_4")


    replanned_trajectory = [(-1.402,2.391), (-1.402,2.391), (-1.402,2.391),(-1.402,2.391),(-1.402,2.391)]
    trajectory_vanilla = [(-1.294, 2.347)]
    for trajectory_node in trajectory_1:
        navigate_to_point(trajectory_node)
        
     
    #rospy.sleep(1)
    #rospy.spin() 

if __name__ == '__main__':
    main()