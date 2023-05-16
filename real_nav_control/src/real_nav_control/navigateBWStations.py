#!/usr/bin/env python3
# license removed for brevity

import rospy
import numpy as np

# Brings in the SimpleActionClient
import actionlib
# Brings in the .action file and messages used by the move base action
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PointStamped
from real_nav_control.srv import *
# from  getDistAngleLidar import getDistAngle
from getDistAngleLidar import getDistAngle

global goalCoord
goalCoord = []

global goalRcvdFlag
goalRcvdFlag =False


def stationCoordClient(stationNo):
    global goalCoord 
    goalCoord = []
    rospy.wait_for_service('stationCoordinates')
    try:
        get_coordinates = rospy.ServiceProxy('stationCoordinates', stationCoord)
        
        coord = get_coordinates(stationNo)

        # print(type(msg))
        
        goalCoord.append(coord.x)
        goalCoord.append(coord.y)
        goalCoord.append(coord.z)
        goalCoord.append(coord.yaw)

        print(type(goalCoord[0]))

        result = movebase_client()

        if result:
            rospy.loginfo("Goal execution done!")        

    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)



# def setGoal(msg):
#     global goalCoord 
#     goalCoord = []
#     goalCoord.append(round(msg.point.x,4))
#     goalCoord.append(round(msg.point.y,4))
#     goalCoord.append(round(msg.point.z,4))

#     result = movebase_client()
#     if result:
#         rospy.loginfo("Goal execution done!")

#     print(msg.point)

def movebase_client():
    global goalRcvdFlag
    print("movebase_client")
   # Create an action client called "move_base" with action definition file "MoveBaseAction"
    client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
 
   # Waits until the action server has started up and started listening for goals.
    client.wait_for_server()

   # Creates a new goal with the MoveBaseGoal constructor
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
   # Move 0.5 meters forward along the x axis of the "map" coordinate frame 
    goal.target_pose.pose.position.x = goalCoord[0]
    goal.target_pose.pose.position.y = goalCoord[1]
    goal.target_pose.pose.position.z = goalCoord[2]
    print(goal.target_pose.pose.position.x)
    print(goal.target_pose.pose.position.y)
    print(goal.target_pose.pose.position.z)
   # No rotation of the mobile base frame w.r.t. map frame
    goal.target_pose.pose.orientation.w = -1

   # Sends the goal to the action server.
    client.send_goal(goal)
   # Waits for the server to finish performing the action.
    wait = client.wait_for_result()
   # If the result doesn't arrive, assume the Server is not available
    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")
    else:
    # Result of executing the action
        # goalRcvdFlag = False
        return client.get_result()   

# def goalLocation():

#     global goalRcvdFlag

#     rospy.init_node('goalLocation', anonymous=True)

#     rospy.Subscriber('/clicked_point' , PointStamped, setGoal)

#     # if goalRcvdFlag== True:
#     #     result = movebase_client()
#     #     if result:
#     #         rospy.loginfo("Goal execution done!")

#     # spin() simply keeps python from exiting until this node is stopped
#     rospy.spin()


# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':
    try:
       # Initializes a rospy node to let the SimpleActionClient publish and subscribe
        rospy.init_node('movebase_client_py')

        while(True):
            print("1 : Workstation 1\n")
            print("2 : Workstation 2\n")
            print("3 : Workstation 3\n")
            print("4 : exit\n\n")

            print("Enter the goal (workstation 1,2 and 3) \n")
            goal = int(input())

            if (goal==1):
                #call workstation 1 service
                print("Goal: workstation 1")
                target_camera_angle = 24  #in degrees

                objectDist = getDistAngle(target_camera_angle)
                print("object distance: ", objectDist)
                stationCoordClient(goal)
                print("Completed")
            elif goal == 2:
                #call workstation 1 service
                print("Goal: workstation 2")
                stationCoordClient(goal)
                print("Completed")
            elif goal == 3:
                #call workstation 1 service
                print("Goal: workstation 3")
                stationCoordClient(goal)
                print("Completed")
            elif goal == 4:
                print("Exit")
                break
            


            
            # goalLocation()

    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")

