import rospy
from nav_msgs.msg import Odometry



def pose_callback(msg):
    print('------------------------------------')
    print('pose x  = ' + str(msg.pose.pose.position.x))

def listener():
    rospy.init_node("odometry node xyz",anonymous=True)
    rospy.Subscriber("/odom", Odometry, pose_callback, queue_size=10)
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass