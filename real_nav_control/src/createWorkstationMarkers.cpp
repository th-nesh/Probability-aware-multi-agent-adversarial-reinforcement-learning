#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

int main( int argc, char** argv )
{
  ros::init(argc, argv, "station1");
  ros::NodeHandle n;
  ros::Rate r(1);
  ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("station1_marker", 1);

  // Set our initial shape type to be a cube
  uint32_t shape = visualization_msgs::Marker::CUBE;

  while (ros::ok())
  {
    visualization_msgs::Marker station1_marker;

    // Set the frame ID and timestamp.  See the TF tutorials for information on these.
    station1_marker.header.frame_id = "map";
    station1_marker.header.stamp = ros::Time::now();

    // Set the namespace and id for this marker.  This serves to create a unique ID
    // Any marker sent with the same namespace and id will overwrite the old one
    station1_marker.ns = "station1";
    station1_marker.id = 0;

    // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    station1_marker.type = shape;

    // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    station1_marker.action = visualization_msgs::Marker::ADD;

    // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
    station1_marker.pose.position.x = 3.673230643998085;
    station1_marker.pose.position.y = 1.3334914424466238;
    station1_marker.pose.position.z = 0.23;
    station1_marker.pose.orientation.x = 0.0;
    station1_marker.pose.orientation.y = 0.0;
    station1_marker.pose.orientation.z = 0.0;
    station1_marker.pose.orientation.w = 1.0;

    // station1_marker.pose.position.x = 4.0022;
    // station1_marker.pose.position.y = 2.5017;
    // station1_marker.pose.position.z = 0.0036;
    // station1_marker.pose.orientation.x = 0.0;
    // station1_marker.pose.orientation.y = 0.0;
    // station1_marker.pose.orientation.z = 0.0;
    // station1_marker.pose.orientation.w = 1.0;


    // Set the scale of the marker -- 1x1x1 here means 1m on a side
    station1_marker.scale.x = 0.1;
    station1_marker.scale.y = 0.1;
    station1_marker.scale.z = 0.1;

    // Set the color -- be sure to set alpha to something non-zero!
    station1_marker.color.r = 0.0f;
    station1_marker.color.g = 1.0f;
    station1_marker.color.b = 0.0f;
    station1_marker.color.a = 1.0;

    station1_marker.lifetime = ros::Duration();

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
    marker_pub.publish(station1_marker);

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
