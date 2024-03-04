# This script is used to publish a marker at the desired location. The script will ask for x and y coordinates from the
# user in the terminal, and then publish the marker at that location. The marker will be a red sphere with a radius of
# 0.1 meters. The marker will be published to the topic /visualization_marker. The marker will be published once and then
# the script will exit.

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def main():
    rospy.init_node('marker_publisher')
    pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "basic_shapes"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = float(raw_input("Enter x coordinate: "))
    marker.pose.position.y = float(raw_input("Enter y coordinate: "))
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    pub.publish(marker)