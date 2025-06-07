import rospy
from sensor_msgs.msg import LaserScan 
from geometry_msgs.msg import Quaternion
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

import numpy as np

class LiDAR_MAP:

    def __init__(self):
        rospy.init_node('lidar_correction', anonymous=True)
        
        self.roll = 0.0
        self.pitch = 0.0
        self.roll_pitch_subscriber = rospy.Subscriber('/imu/roll_pitch', Quaternion, self.roll_pitch_callback)

        # Get rosparameters scan_input and scan_output
        self.scan_input = rospy.get_param('~scan_input', '/scan')
        self.scan_output = rospy.get_param('~scan_output', '/scan_corrected')

        self.scan_publisher = rospy.Publisher(self.scan_output, LaserScan, queue_size=10)
        self.scan_subscriber = rospy.Subscriber(self.scan_input, LaserScan, self.scan_callback, queue_size=1)


        # Initialize static transform broadcaster
        self.tf_broadcaster = rospy.Publisher('/tf_static', TFMessage, queue_size=10)

        # Create a static transform from map to odom
        self.map_to_odom = TransformStamped()
        self.map_to_odom.header.frame_id = "map"
        self.map_to_odom.child_frame_id = "odom"
        self.map_to_odom.transform.translation.x = 0.0
        self.map_to_odom.transform.translation.y = 0.0
        self.map_to_odom.transform.translation.z = 0.0
        self.map_to_odom.transform.rotation.x = 0.0
        self.map_to_odom.transform.rotation.y = 0.0
        self.map_to_odom.transform.rotation.z = 0.0
        self.map_to_odom.transform.rotation.w = 1.0

        # Publish the static transform once
        tf_msg = TFMessage([self.map_to_odom])
        self.tf_broadcaster.publish(tf_msg)

        # Set up a timer to periodically publish the transform
        self.tf_timer = rospy.Timer(rospy.Duration(1.0), self.publish_tf)

    def run(self):
        rospy.spin()

    def publish_tf(self, event):
        # Update the timestamp for the transform
        self.map_to_odom.header.stamp = rospy.Time.now()
        
        # Publish the static transform
        tf_msg = TFMessage([self.map_to_odom])
        self.tf_broadcaster.publish(tf_msg)

    def roll_pitch_callback(self, roll_pitch_data):
        # Convert Quaternion to roll and pitch
        (self.roll, self.pitch, _) = euler_from_quaternion([
            roll_pitch_data.x,
            roll_pitch_data.y,
            roll_pitch_data.z,
            roll_pitch_data.w
        ])

        # print(f"Roll: {self.roll}, Pitch: {self.pitch}")

    def scan_callback(self, scan_data):

        if np.abs(self.roll) < 0.017 and np.abs(self.pitch) < 0.017: # (1 degree)
            # If roll and pitch are negligible, just publish the original scan
            self.scan_publisher.publish(scan_data)
        else: 
            # Extract ranges and angles from the scan data
            scan_ranges = np.array(scan_data.ranges)
            
            angle_min = scan_data.angle_min
            angle_increment = scan_data.angle_increment
            angle_max = scan_data.angle_max
            scan_angles = np.arange(angle_min, angle_max, angle_increment)

            # 1. Convert polar to Cartesian coordinates
            points = np.vstack([
                scan_ranges * np.cos(scan_angles),  # x
                scan_ranges * np.sin(scan_angles),  # y
                np.zeros_like(scan_ranges)          # z
            ])

            # 2. Create combined rotation matrix (roll then pitch)
            cp, sp = np.cos(self.pitch), np.sin(self.pitch)
            cr, sr = np.cos(self.roll), np.sin(self.roll)
            
            # Inverse (transpose) or R_pitch * R_roll rotation matrices
            rotation_matrix = np.array([
                [cp,          0,     -sp],
                [sr*sp,       cr,   sr*cp],
                [cr*sp,      -sr,   cr*cp]
            ])

            # 3. Apply the rotation to the scan data
            rotated_points = rotation_matrix @ points

            # 4. Project to z=0 plane
            projected_points = rotated_points[:2, :]
            corrected_ranges = np.linalg.norm(projected_points, axis=0)
            corrected_angles = scan_angles

            # 5. Create a new LaserScan message
            corrected_scan = LaserScan()
            corrected_scan.header = scan_data.header
            corrected_scan.angle_min = angle_min
            corrected_scan.angle_max = angle_max    
            corrected_scan.angle_increment = angle_increment
            corrected_scan.time_increment = scan_data.time_increment
            corrected_scan.range_min = scan_data.range_min
            corrected_scan.range_max = scan_data.range_max
            corrected_scan.ranges = corrected_ranges.tolist()
            corrected_scan.intensities = scan_data.intensities

            # 6. Publish the corrected scan
            self.scan_publisher.publish(corrected_scan)

if __name__ == '__main__':
    try:
        lidar_map = LiDAR_MAP()
        lidar_map.run()
    except rospy.ROSInterruptException:
        pass

