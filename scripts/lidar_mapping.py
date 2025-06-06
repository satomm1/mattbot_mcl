import rospy
from sensor_msgs.msg import LaserScan 
from geometry_msgs.msg import Quaternion
from tf.transformations import euler_from_quaternion

import numpy as np

class LiDAR_MAP:

    def __init__(self):
        rospy.init_node('lidar_mapping', anonymous=True)
        
        self.roll = 0.0
        self.pitch = 0.0
        self.roll_pitch_subscriber = rospy.Subscriber('/imu/roll_pitch', Quaternion, self.roll_pitch_callback)

        self.scan_publisher = rospy.Publisher('/scan_corrected', LaserScan, queue_size=10)
        self.scan_subscriber = rospy.Subscriber('/scan', LaserScan, self.scan_callback)

    def run(self):
        rospy.spin()

    def roll_pitch_callback(self, roll_pitch_data):
        # Convert Quaternion to roll and pitch
        (self.roll, self.pitch, _) = euler_from_quaternion([
            roll_pitch_data.x,
            roll_pitch_data.y,
            roll_pitch_data.z,
            roll_pitch_data.w
        ])

    def scan_callback(self, scan_data):
        # Process the scan data and publish it
        # Here you can implement your mapping logic
        self.scan_publisher.publish(scan_data)
        rospy.loginfo("Published scan data to /lidar_map")

        scan_depths = np.array(scan_data.ranges)
        
        angle_min = scan_data.angle_min
        angle_increment = scan_data.angle_increment
        angle_max = scan_data.angle_max
        scan_angles = np.arange(angle_min, angle_max, angle_increment)

        # Correct the scan data based on roll and pitch
        corrected_depths = scan_depths * np.cos(self.roll) * np.cos(self.pitch)

if __name__ == '__main__':
    try:
        lidar_map = LiDAR_MAP()
        lidar_map.run()
    except rospy.ROSInterruptException:
        pass

