#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

int main( int argc, char** argv )
{
  ros::init(argc, argv, "station3");
  ros::NodeHandle n;
  ros::Rate r(1);
  ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("station3_marker", 1);

  // Set our initial shape type to be a cube
  uint32_t shape = visualization_msgs::Marker::CUBE;

  while (ros::ok())
  {
    visualization_msgs::Marker station3_marker;
    // Set the frame ID and timestamp.  See the TF tutorials for information on these.
    station3_marker.header.frame_id = "map";
    station3_marker.header.stamp = ros::Time::now();

    // Set the namespace and id for this marker.  This serves to create a unique ID
    // Any marker sent with the same namespace and id will overwrite the old one
    station3_marker.ns = "station3";
    station3_marker.id = 0;

    // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    station3_marker.type = shape;

    // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    station3_marker.action = visualization_msgs::Marker::ADD;

    // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
    station3_marker.pose.position.x = 4.2366743087768555;
    station3_marker.pose.position.y = 0.41317152976989746;
    station3_marker.pose.position.z = 0.0021371841430664062;
    station3_marker.pose.orientation.x = 0.0;
    station3_marker.pose.orientation.y = 0.0;
    station3_marker.pose.orientation.z = 0.0;
    station3_marker.pose.orientation.w = 1.0;

    // Set the scale of the marker -- 1x1x1 here means 1m on a side
    station3_marker.scale.x = 0.5;
    station3_marker.scale.y = 0.5;
    station3_marker.scale.z = 0.5;

    // Set the color -- be sure to set alpha to something non-zero!
    station3_marker.color.r = 0.0f;
    station3_marker.color.g = 1.0f;
    station3_marker.color.b = 0.0f;
    station3_marker.color.a = 1.0;

    station3_marker.lifetime = ros::Duration();

    // Publish the marker
    while (marker_pub.getNumSubscribers() < 1)
    {
      if (!ros::ok())
      {
        return 0;
      }
      ROS_WARN_ONCE("Please create a subscriber to the marker");
      sleep(1);
    }
    marker_pub.publish(station3_marker);

    // Cycle between different shapes
//    switch (shape)
//    {
//    case visualization_msgs::Marker::CUBE:
//      shape = visualization_msgs::Marker::SPHERE;
//      break;
//    case visualization_msgs::Marker::SPHERE:
//      shape = visualization_msgs::Marker::ARROW;
//      break;
//    case visualization_msgs::Marker::ARROW:
//      shape = visualization_msgs::Marker::CYLINDER;
//      break;
//    case visualization_msgs::Marker::CYLINDER:
//      shape = visualization_msgs::Marker::CUBE;
//      break;
//    }

    r.sleep();
  }
}
