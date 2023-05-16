
import cv2
import numpy as np
from conversions import polar2Cartesion
import rospy
from geometry_msgs.msg import Twist, Vector3
from pyzbar.pyzbar import decode
from math import pi
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
real_data =[]

#Global variable to read camera data
Image_data = []


#Global counter to save image
image_counter = 0   

#Global datcollection
target_identified = {}
objectDist_camera=0
object_y_camera= 0
class example:
    staticVariable = True # Access through class

def callback(data):
    now = rospy.Time.now()
    # #while rospy.Time.now()<now+rospy.Duration.from_sec(1):
    if(example.staticVariable):
        #print('camera called /n-------------------------------------')
        global image_counter
        global target_identified
        global objectDist_camera
        global object_y_camera
        image_counter += 1
    
        br = CvBridge()

        rospy.logdebug("Recieving video frames")

        img = br.imgmsg_to_cv2(data, "rgb8")
        
        width = int(data.width * 0.80)
        height = int(data.height * 0.80)
        dim = (width, height)

        img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        if ((decode(img_resized))):
            
            barcode_image = 0
        # print("DETECTING QR CODE---------------------------------------------------------------------------")
            for barcode in decode(img_resized):

                #intilization of the cummulative value 
                center_x_value =0
                center_y_value = 0
                width_actual = 0 
                height_actual = 0
                widthbound = 0
                heightbound = 0

                barcode_image = barcode_image+1
                
                for iter in range(2):

                    # decode the bar code ex wokrstation1 
                    mydata = barcode.data.decode('utf-8')

                    # print(barcode.data)
                    pts = np.array([barcode.polygon],np.int32)
                    pts = pts.reshape(-1,1,2)
                    cv2.polylines(img_resized,[pts],True,(255,0,255),5)
                    pts2 = barcode.rect

                    #Calculate the center of the QR code
                    center_x = pts2[0]+(pts2[2]/2)
                    center_y = pts2[1]+(pts2[3]/2)

                    #cummulative value of centre over 250 iteration 
                    center_x_value = center_x + center_x_value
                    center_y_value = center_y + center_y_value

                    # cummulaitve value of bounding box width and height
                    widthbound = widthbound+ pts2[2]
                    heightbound = heightbound+pts2[3]



            #mean value of all cummulative 
            widthbound = widthbound
            heightbound = heightbound
            center_x_value = center_x_value
            center_y_value = center_y_value

            

            #Calculate the angle of the target wrt to camera
            camera_angle_xdirection =  36 - (0.1125*(center_x_value))
            camera_angle_ydirection =  36 - (0.1125*(center_y_value))
            camera_angle = (camera_angle_xdirection+camera_angle_ydirection)/2


            Distance_demo= 1.22
            Width_Demo = 96

            #formula for calculating new distance using known parameter
            objectDist_camera =(Distance_demo * Width_Demo  )/( widthbound) 


            #to check the distance is less between the scanner and robotino
            if objectDist_camera<2.0 and objectDist_camera!= 0.0:
                # Find the x and y axis of the target wrt to camera frame
                object_x_camera, object_y_camera = polar2Cartesion(objectDist_camera,camera_angle)
                x_value = -0.967-(object_x_camera*-0.625)
                y_value = 0.2-(object_y_camera*-0.9)
               # print('----------x---------',x_value)
                #print('--------y----value',y_value)       
               


def rotate(t,sign1=1):
    velocity_publisher_robot = rospy.Publisher('cmd_vel_real', Twist, queue_size=1)
    move = Twist()
    move.linear.x = 0 
    move.linear.y = 0
    move.angular.z = sign1*0.1
    now = rospy.Time.now()
    rate = rospy.Rate(10)

    while rospy.Time.now()<now+rospy.Duration.from_sec(t):
        velocity_publisher_robot.publish(move)
        #rate.sleep()

def move(t):
    velocity_publisher_robot = rospy.Publisher('cmd_vel_real', Twist, queue_size=1)
    move = Twist()
    move.linear.x = 0.1 
    move.linear.y = 0
    move.angular.z = 0
    now = rospy.Time.now()
    rate = rospy.Rate(10)

    while rospy.Time.now()<now+rospy.Duration.from_sec(t):
        velocity_publisher_robot.publish(move)
        #rate.sleep()

def main():

    rospy.init_node('Swapnil_test', anonymous=True)
    #rospy.Subscriber('/image_raw', Image, callback, queue_size=10)
    arrroate=[10,3, 13,12,22] # ,12,9
    arrmove=[14,0, 12,0,14] # ,12,14
    sign1 = [1,1,-1,1,-1] # -1
    for r,m,s,i in zip(arrroate,arrmove,sign1, range(len(arrroate))):
        print(r, m, i)
        rotate(r,s)
        move(m)
        if (i+1) % 2 == 0:
            print("in camera")
            now = rospy.Time.now()
            rate = rospy.Rate(0.5)
            sub = rospy.Subscriber('/image_raw', Image, callback, queue_size=10)
            rate.sleep()
            sub.unregister()
            if object_y_camera>-0.3:
                arrroate[i+1]=arrroate[i]+1
                #arrmove[i+1]=arrmove[i]+1
                print('------changed-----',arrroate[i+1])
                print('------changed-----',arrmove[i+1])
                print(object_y_camera)

            
            #rate.sleep()
        

if __name__ == '__main__':
    main()  