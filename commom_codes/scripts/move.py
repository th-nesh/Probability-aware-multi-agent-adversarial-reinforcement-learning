import rospy

from geometry_msgs.msg import Twist

def drive_forward():
    msg_drive_forward = Twist()

    msg_drive_forward.linear.x = 1.5
    msg_drive_forward.linear.y = 0
    msg_drive_forward.linear.z = 0
    msg_drive_forward.angular.x = 0
    msg_drive_forward.angular.y = 0
    msg_drive_forward.angular.z = 0
    print('move_forward')
    return msg_drive_forward

def move_forward():
    real_velocity_publisher = rospy.Publisher('/cmd_vel_real',Twist,queue_size=10)
    real_velocity_publisher.publish(drive_forward())

def drive_backward():
    msg_drive_backward = Twist()

    msg_drive_backward.linear.x = -1.5
    msg_drive_backward.linear.y = 0
    msg_drive_backward.linear.z = 0
    msg_drive_backward.angular.x = 0
    msg_drive_backward.angular.y = 0
    msg_drive_backward.angular.z = 0
    print('move_backward')
    return msg_drive_backward

def move_backward():
    real_velocity_publisher = rospy.Publisher('/cmd_vel_real',Twist,queue_size=10)
    real_velocity_publisher.publish(drive_backward())

if __name__ == "__main__":
    rospy.init_node('auto_drive', anonymous=True)
    while not rospy.is_shutdown():
        #move_forward()
        move_backward()
        
