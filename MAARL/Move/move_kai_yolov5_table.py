
import random
import numpy as np
import time
import copy
#from PIL import Image, ImageDraw
from PIL import Image, ImageDraw
import sys
import os
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
from object_detection import Obstacle
from PRM import apply_PRM, apply_PRM_init, draw_traj, calc_adv_traj, Node, get_traj_edges, get_node_with_coordinates, calc_nearest_dist
from object_detection import apply_object_detection
from copy import deepcopy
from object_detection import Obstacle
#sys.path.insert(0, os.path.dirname(os.path.dirname(__file__))+'/yolov7')
# from detect_online import get_conf_and_model, loaded_detect

real_data =[]
image_counter = 0
velocity_publisher_robot = rospy.Publisher('cmd_vel_real', Twist, queue_size=10)
deviation =0
center = 0
outer_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
model = torch.hub.load(outer_folder+'/yolov5/yolov5','custom', path=outer_folder+'/yolov5/yolov5/best.pt', force_reload=True, source= "local")
class example:
    staticVariable = True # Access through class


real_data =[]
def initialize_traj(map_ref, obstacles=None, nodes=None,start_node= None, goal_node = None, visualize=False, edges_all=None, env=None):
    """
    initialize a new usually random trajectory for training
    if we pass nodes here then those nodes are used to calculate a trajectory.
    if no nodes are passed then new nodes are generated (takes time)
    @param map_ref: map that we want to use PRM on - usually the reference map
    @param obstacles: known obstacles (list of Obstacles-objects)
    @param nodes: current nodes (for the graph for PRM)
    @param visualize: determines if the new trajectory should be saved as an image (usually for debugging..)
    @param edges_all: edges_all are the current edges of our graph with their respective costs and nodes. They will
                        be calculated new if we calculate new nodes
    @param env: Environment object. Only needed when dynamically generating random maps
    @return: trajectory - new trajectory as list of nodes
             nodes - new or old nodes, depending on param nodes
             edges_all - edges with their new costs (or unchanged costs, depending on whether we passed nodes)
    """
    #import pdb; pdb.set_trace()
    if not nodes:
        if env:
            # env.map_ref, env.obstacles = create_random_map()
            traj, traj_opt, nodes, edges_all = apply_PRM_init(env.map_ref, env.obstacles)
        else:
            traj, traj_opt, nodes, edges_all = apply_PRM_init(map_ref, obstacles)
        
        # pickle dump ~~~
        # print('dumping nodes...')
        # open_file = open('nodes_presentation', "wb")
        # pickle.dump(nodes, open_file)
        # open_file.close()
    elif start_node and goal_node:
        traj, _, nodes, _ = apply_PRM(map_ref, nodes, start_node=start_node, goal_node=goal_node)

    else:
        # for specific start / goal location: ------------------
        # start_node = get_node_with_coordinates(nodes, (62, 74))
        # goal_node = get_node_with_coordinates(nodes, (109, 125))
        # traj, _, nodes, _ = apply_PRM(map_ref, nodes, visualize=visualize, start_node=start_node, goal_node=goal_node)
        # ------------------------------------------------------

        traj, _, nodes, _ = apply_PRM(map_ref, nodes, visualize=visualize)
    # print('fresh trajectory:', traj)

    return traj, nodes, edges_all

def initialize_map(map_path):
    """
    initializes the reference map by applying the object detection
    @param map_path: path for the scanner map
    @return: reference map and the detected obstacles (with its borders)
    """
    map_ref, obstacles = apply_object_detection(map_path)

    # --- only for test we use the already processed map ---
    # obstacles = None
    # ref_map_path = 'map_ref.png'
    # map_ref = Image.open(ref_map_path)
    #obstacle_adversary(obstacles,2)
  

    return map_ref, obstacles

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

    speed = 0.10
    
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
    # now = rospy.Time.now()
    if(example.staticVariable):
        global Image_data
        global image_counter
        #global model
        #global deviation
        global center
        global conf
        global img_glob
        image_counter += 1
        
        bridge = CvBridge()

        cv_image = bridge.imgmsg_to_cv2(img_data,"rgb8")

        #img_glob = deepcopy(cv_image)
        #Resize Image to 640 * 480 - YOLO was trained in this size
        # width = int(img_data.width * 0.80)
        # height = int(img_data.height * 0.80)
        # dim = (width, height)
        # img_resized = cv2.resize(cv_image, dim, interpolation = cv2.INTER_AREA)
        # loaded_detect(cv_image,*conf)
        
        # results =model(img_resized) #model is applied to the camera images
        # labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        # out = plot_boxes(results,img_resized) #plot the bounding boxes to the images
        # print(center)
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
        
        # cv2.imshow('Camera', img_glob) 
        #if image_counter %5 == 0:
            #print(image_counter)
            #Save_Camera_Input = '/home/varthini/catkin_ws/src/IDT/Camera/Images/Test15bck' + str(image_counter) + '.png'
            #cv2.imwrite(Save_Camera_Input, img_resized)


        # cv2.waitKey(1)
        # Resize Image to 640 * 480 - YOLO was trained in this size
        width = int(img_data.width * 0.80)
        height = int(img_data.height * 0.80)
        dim = (width, height)
        img_resized = cv2.resize(cv_image, dim, interpolation = cv2.INTER_AREA)
        img_glob = deepcopy(img_resized)

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
            if row[4] >= 0.5:
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

Pose_real_data = [0,0,0,0]

def convert_robo_to_grid(x_r,y_r):
    "applies the roation of -69° to the robo cords to get the crods in the grind we measure on"
    x_g = x_r*np.cos(-1.204)-y_r*np.sin(-1.204)
    y_g = x_r*np.sin(-1.204)+y_r*np.cos(-1.204)
    return x_g,y_g

def get_pixel_location_from_acml(x_a,y_a,x3,y3,x5_3,x5_3_p,y5_3,y5_3_p):
    "get pixel location from acml pos"
    x_convert = x5_3_p/x5_3 #pixel per meter in width, pixels have flipped x axis (is also offset, applied below)
    y_convert = y5_3_p/y5_3 #pixel per meter in hight, pixels have flipped y axis (is also offset, applied below)
    offset_x = -x3*x_convert+100 # calculating x offset based on the pixel location of x3
    offset_y = y3*x_convert+150 # calculating y offset based on the pixel location of y3
    x_p = x_a*x_convert+offset_x # calulating pixel value of x
    y_p = y_a*y_convert+offset_y # calulating pixel value of y
    return x_p, y_p

def get_amcl_from_pixel_location(x_p,y_p,x3,y3,x5_3,x5_3_p,y5_3,y5_3_p):
    "get pixel location from acml pos"
    x_convert = x5_3_p/x5_3 #pixel per meter in width, pixels have flipped x axis (is also offset, applied below)
    y_convert = y5_3_p/y5_3 #pixel per meter in hight, pixels have flipped y axis (is also offset, applied below)
    offset_x = -x3*x_convert+100 # calculating x offset based on the pixel location of x3
    offset_y = y3*x_convert+150 # calculating y offset based on the pixel location of y3
    #x_p = x_a*x_convert+offset_x # calulating pixel value of x
    #y_p = y_a*y_convert+offset_y # calulating pixel value of y
    x_a = (x_p - offset_x)/ x_convert
    y_a = (y_p- offset_y)/y_convert
    return x_a, y_a

def show_acml_as_pixel():
    #Workstation_5
    #position: 
    x5= -2.4831574505182648
    y5= 2.9521294984392905
    #orientation: 
    z2= 0.9865593763339435
    w1= 0.16340317306460247
    x5_p = 65
    y5_p = 91
    #workstation_3
    #position: 
    x3= -0.8142565856271811
    y3= 0.1506975961760537
    #orientation: 
    z1= -0.8168027744144787
    w1= 0.5769170024438613
    x3_p = 100
    y3_p = 150
    # workstation_4
    x4 = -1.95
    y4 = 0.552

    # distance to check center point
    y5_3 = y5-y3
    x5_3 = x5-x3
    y5_3_p = y5_p-y3_p
    x5_3_p = x5_p-x3_p
    print(x5_3, y5_3,' distance between ws3 and ws5 in robo cords')
    # now in tile coordinates
    x_mes, y_mes = convert_robo_to_grid(x5_3, y5_3)
    x4_3, y4_3 = convert_robo_to_grid(x4-x3, y4-y3)
    x_from_0,y_from_0 = convert_robo_to_grid(x3, y3)
    print(x_from_0,y_from_0, 'distance between ws3 and 0,0 in grid cords')
    print(x_mes, y_mes, 'should be 2.07 2.56  distance between ws3 and ws5 in grid cords')
    print(x4_3, y4_3, 'distance between ws3 and ws4 in grid cords')


    # Unit conversion pose to pixel, the pixels have a fliped y axis
    print(x5_3/(65-100) , ' meters per pixel in width')
    print(-y5_3/(91-150) , ' meters per pixel in hight')

    x_p, y_p = get_pixel_location_from_acml(x5,y5,x3,y3,x5_3,x5_3_p,y5_3,y5_3_p)
    print(x_p, y_p, 'pixel values ws 5, should be 65, 91')
    x_p, y_p = get_pixel_location_from_acml(x3,y3,x3,y3,x5_3,x5_3_p,y5_3,y5_3_p)
    print(x_p, y_p, 'pixel values ws 3, should be 100, 150')
    x_p, y_p = get_pixel_location_from_acml(x4,y4,x3,y3,x5_3,x5_3_p,y5_3,y5_3_p)
    print(x_p, y_p, 'pixel values ws 4, should be 75, 142')

    while not rospy.is_shutdown():
        #  pixel_location = get_pixel_location_from_acml(Pose_real_data[1],Pose_real_data[2],x3,y3,x5_3,x5_3_p,y5_3,y5_3_p)
         pixel_location = get_pixel_location_from_acml(real_data[1],real_data[2],x3,y3,x5_3,x5_3_p,y5_3,y5_3_p)
         print(pixel_location, end = "\r")
def get_base_info():
        #Workstation_5
    #position: 
    x5= -2.4831574505182648
    y5= 2.9521294984392905
    #orientation: 
    z2= 0.9865593763339435
    w1= 0.16340317306460247
    x5_p = 65
    y5_p = 91
    #workstation_3
    #position: 
    x3= -0.8142565856271811
    y3= 0.1506975961760537
    #orientation: 
    z1= -0.8168027744144787
    w1= 0.5769170024438613
    x3_p = 100
    y3_p = 150
    # workstation_4
    x4 = -1.95
    y4 = 0.552

    # distance to check center point
    y5_3 = y5-y3
    x5_3 = x5-x3
    y5_3_p = y5_p-y3_p
    x5_3_p = x5_p-x3_p
    return x3,y3,x5_3,x5_3_p,y5_3,y5_3_p

def convert_robo_to_cam(x_r,y_r, rot):
    "applies the roation of -69° to the robo cords to get the crods in the grind we measure on"
    x_g = x_r*np.cos(rot)-y_r*np.sin(rot)
    y_g = x_r*np.sin(rot)+y_r*np.cos(rot)
    return x_g,y_g

def distance_to_ws(bot,top):
    global real_data
    dist_bot = bot
    dist_top = top
    dist_bot = (bot[0]-real_data[1], bot[1]-real_data[2])
    dist_top = (top[0]-real_data[1], top[1]-real_data[2])
    rot = -2*np.arcsin(real_data[3])
    #rot_acml = 
    # if real_data[3]>0:
    #     rot = -1.204-(real_data[3]-0.9831301592470502)*np.pi
    #     rot = np.arcsin(real_data[3]-0.9831301592470502)
    #     if rot<0:
    #         rot= 2*np.pi+rot
    # else:
    #     rot = -1.204-((2-real_data[3])-0.9831301592470502)*np.pi
    #     rot = np.arcsin(1+real_data[3])+np.arcsin(1-0.9831301592470502)

    # print('distance to bot/top corner',convert_robo_to_cam(dist_bot[0],dist_bot[1],rot),convert_robo_to_cam(dist_top[0],dist_top[1],rot))
    # print(2*np.arcsin(real_data[3]), end = '\r')
    # print('distance to top corner',convert_robo_to_cam(dist_top[0],dist_top[1],rot))
    return convert_robo_to_cam(dist_bot[0],dist_bot[1],rot),convert_robo_to_cam(dist_top[0],dist_top[1],rot)

def camera_to_acml_roation(dist):
    global real_data
    rot = 2*np.arcsin(real_data[3])
    return convert_robo_to_cam(dist[0],dist[1],rot)

def infer_depth(top,bot,h_real):
    hight = top-bot
    a = 651.6301600000004*h_real/0.8
    return a/np.abs(hight)

def infer_depth_from_top(top,h_real):
    offset = 278.9999553571429
    hight = 600-top-285.0887749553571
    a = 285.0887749553571*(h_real)/0.8
    return a/np.abs(hight)

def infer_width(shift_p, depth):
    a_y = 0.0012277075719758462
    offset_y = 400
    return (shift_p-offset_y)*a_y*depth
        
def obstacle_adversary(obstacles,action_index):
    """
    initializes the reference map by applying the object detection
    @param obstacles: obstacles in the map
    @param action_index: action of the adversary
    @return: modified obstacles
    """
    obstacles_disturbed =  obstacles
    #print(obstacles_disturbed)
    for o in range(0,len(obstacles)):
        for i in range(0,len(obstacles[o].corners)):
            obstacles_disturbed[o].corners[i] = tuple(map(lambda j,k : j + k, obstacles_disturbed[o].corners[i], action_index))
        for i in range(0,len(obstacles[o].borders)):
            obstacles_disturbed[o].borders[i][0] = tuple(map(lambda j,k : j + k*np.cos(0), obstacles_disturbed[o].borders[i][0], action_index))
            obstacles_disturbed[o].borders[i][1] = tuple(map(lambda j,k : j + k*np.cos(0), obstacles_disturbed[o].borders[i][1], action_index))
    #print(obstacles_disturbed)
    return obstacles_disturbed

def main():
    global conf
    global model
    global img_glob
    #conf = get_conf_and_model()
    
      
    map_path = "/home/kaib/catkin_ws/src/FinalScannerMap1.png"
    map_ref, obstacles = initialize_map(map_path)
    # # Hand insert
    # corners_hand = [(60,102),(69,80),(57,75),(47,94)]
    # obst_hand = Obstacle(corners_hand)
    # obstacles.append(obst_hand)

    # trajectory_vanilla, nodes_vanilla, edges_all_vanilla = initialize_traj(map_ref,obstacles, nodes=None)
    #import pdb; pdb.set_trace()
    # print(trajectory_vanilla)
    rospy.init_node('test_thinesh', anonymous=True)
    while not rospy.is_shutdown():
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, realCallback, queue_size=10)
        rospy.sleep(1)
        #deviation =0
        
        #trajectory_0= [(-0.7153,2.048), (-1.402,2.391), (-2.062,2.018)]
        #trajectory_1= [ (-0.9501,0.969),(-1.645,1.44), (-2.062,2.018)]
        #trajectory_2= [ (-0.885,1.196), (-2.062,2.018)]
        #trajectory_3= [(-0.7153,2.048), (-1.38,2.196),(-2.062,2.018)]
        #trajectory_4= [ (-2.062,2.018)]
        #print("in camera")
        # now = rospy.Time.now()
        # rate = rospy.Rate(0.5)
        sub = rospy.Subscriber('/image_raw', Image, Image_Callback, queue_size=10)
        # rate.sleep()
        #sub.unregister()
        """if deviation == 0:
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
                    print("trajectory_4")"""""
        # trajectory = [(142.85,78.7),(125.38,70.15),(88.3,73.03)]
        # traj_updt = []
        # base_info =  get_base_info()
        # for i in range(len(trajectory_vanilla)):
        #     traj =  get_amcl_from_pixel_location(trajectory_vanilla[i].x,trajectory_vanilla[i].y,*base_info)
        #     traj_updt.append(traj)
        # print(traj_updt)
        # visu_adv_traj_map = copy.deepcopy(map_ref)
        # visu_adv_traj_map = visu_adv_traj_map.convert('RGB')
        # visu_adv_traj_map_draw = ImageDraw.Draw(visu_adv_traj_map)
        # for i in range(0,len(trajectory_vanilla)-1):
        #     visu_adv_traj_map_draw.line([(trajectory_vanilla[i].coordinates[0], trajectory_vanilla[i].coordinates[1]), (trajectory_vanilla[i+1].coordinates[0],trajectory_vanilla[i+1].coordinates[1])], fill=(255, 255, 0))
        #     visu_adv_traj_map_draw.point([(trajectory_vanilla[i].coordinates[0], trajectory_vanilla[i].coordinates[1])], fill=(200, 255, 0))
        #     visu_adv_traj_map_draw.point([(trajectory_vanilla[i+1].coordinates[0], trajectory_vanilla[i+1].coordinates[1])], fill=(200, 255, 0))
                
        
        # try:
        #             visu_adv_traj_map.save('./image/adv_trajectory.png')
        #             visu_adv_traj_map.save('./image/adv_trajectory_DEBUG.png')
        # except PermissionError:
        #             print('permissionError when saving file')
        rospy.sleep(1)

        od_top = [0,0,0]
        od_bot = [0,0,0]
        #im0 = cv2.imread('./yolov7/data/bb/img1.png')
        # while not rospy.is_shutdown():
        #     bot = obstacles[3].corners[0]
        #     top = obstacles[3].corners[1]
        #     bot_amcl =  get_amcl_from_pixel_location(bot[0],bot[1],*base_info)
        #     top_acml =  get_amcl_from_pixel_location(top[0],top[1],*base_info)
        #     bot_amcl,top_acml = distance_to_ws(bot_amcl,top_acml)
        #     img_local = deepcopy(img_glob)
        #     od_bot_tmp, od_top_tmp = loaded_detect(img_local,*conf)
        #     if od_top_tmp is not None:
        #         od_top = od_top_tmp
        #         od_bot = od_bot_tmp
        #     print(bot_amcl[0]-od_bot[2]+0.15,bot_amcl[1]+od_bot[0],top_acml[0]-od_top[2]+0.15,top_acml[1]+od_top[0],'accuracy')
        #     #time.sleep(1)
        base_inf = get_base_info()
        while not rospy.is_shutdown():
            map_numpy = cv2.imread(map_path)
            img_local = deepcopy(img_glob)
            results =model(img_local) #model is applied to the camera images
            results.xyxy[0] = results.xyxy[0].cpu()
            results.xyxyn[0] = results.xyxyn[0].cpu()
            labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
            #print(cord)
            list_of_depth = []
            list_of_width = []
            for obj,label in zip(cord,labels):
                if label == 2:
                    global real_data
                    list_of_depth.append(infer_depth(obj[1]*600,obj[3]*600,0.74))
                    list_of_depth.append((infer_width(obj[0]*800,list_of_depth[0]),infer_width(obj[2]*800,list_of_depth[0])))

                    

                    print(type(map_numpy))

                    y = list_of_depth[0]
                    x = list_of_depth[1][0]
                    # distance in meters rotated into acml_pose
                    shift_in_acml = camera_to_acml_roation((y,-x))
                    # location in acml
                    left_in_acml = shift_in_acml[0]+real_data[1],shift_in_acml[1]+real_data[2]
                    # location in pixels
                    left_corner = get_pixel_location_from_acml(*left_in_acml,*base_inf)
                    y = list_of_depth[0]
                    x = list_of_depth[1][1]
                    # distance in meters rotated into acml_pose
                    shift_in_acml = camera_to_acml_roation((y,-x))
                    # location in acml
                    right_in_acml = shift_in_acml[0]+real_data[1],shift_in_acml[1]+real_data[2]
                    # location in pixels
                    right_corner = get_pixel_location_from_acml(*right_in_acml,*base_inf)

                    curr_loc = get_pixel_location_from_acml(real_data[1], real_data[2],*base_inf)
                       # # Hand insert

                    
                    corners_hand = [(98,85),(110,85),(114,73),(107,65)]
                    obst_hand = Obstacle(corners_hand)
                    visu_adv_traj_map = copy.deepcopy(map_ref)
                        
                    visu_adv_traj_map = visu_adv_traj_map.convert('RGB')
                    visu_adv_traj_map_draw = ImageDraw.Draw(visu_adv_traj_map)
                    
                    for j in range(0, len(obst_hand.corners)):
                        visu_adv_traj_map_draw.point([(obst_hand.corners[j])], fill=(255, 0, 0))

                    for j in range(0, len(obst_hand.borders)):
                        visu_adv_traj_map_draw.line([obst_hand.borders[j][0], obst_hand.borders[j][1]], fill=(0, 255, 0))
                    shift_pix = (left_corner[0]-corners_hand[0][0],left_corner[1]-corners_hand[0][1])

                    shifted_obst = obstacle_adversary([obst_hand],shift_pix)[0]

                    for j in range(0, len(shifted_obst.corners)):
                        visu_adv_traj_map_draw.point([(shifted_obst.corners[j])], fill=(255, 0, 0))

                    for j in range(0, len(shifted_obst.borders)):
                        visu_adv_traj_map_draw.line([shifted_obst.borders[j][0], shifted_obst.borders[j][1]], fill=(0, 0, 255))
                    

                    cv2.line(map_numpy, np.array(left_corner).astype(int), np.array(right_corner).astype(int), [random.randint(0, 255) for _ in range(3)], thickness=2, lineType=cv2.LINE_AA)
                    
                    print('location in pixels for left/right',left_corner[0],left_corner[1],right_corner[0],right_corner[1])
                    # print('location in pixels for left/right',left_corner[0]+curr_loc[0],left_corner[1]+curr_loc[1],right_corner[0]+curr_loc[0],right_corner[1]+curr_loc[1])
                    # print(curr_loc)

            print('depths', list_of_depth)
            out = plot_boxes(results,img_local) #plot the bounding boxes to the images   
            cv2.imshow('map', np.asarray(visu_adv_traj_map))
            cv2.imshow('Camera', out)
            cv2.waitKey(1)         
            #img_local = deepcopy(img_loaded_detect(img_local,*conf) glob)
            #loaded_detect(img_local,*conf)

            # print('test')loaded_detect(img_local,*conf) 
            # navigate_to_point(trajectory_node)

            
        
        #rospy.sleep(1)
        #rospy.spin() 

if __name__ == '__main__':
    main()