import rospy

from sensor_msgs.msg import LaserScan

import numpy as np

class LidarCorrection:

    def __init__(self):
        rospy.init_node('lidar_correction', anonymous=True)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)  # queue size 1 to avoid processing old messages since no longer relevant

    def scan_callback(self, msg):
        
        t1 = rospy.Time.now()
        # Adjust the data based on a roll of 0.1 radians
        roll_correction = 0.1
        laser_scans = np.array(msg.ranges)

        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        angle_max = msg.angle_max

        angles = np.arange(angle_min, angle_max, angle_increment)

        # Calculate the correction for each angle based on the roll
        # Assuming roll_correction is a constant offset for simplicity
        # In a real scenario, you might want to calculate this based on the angle
        # Here we just apply a constant correction for demonstration purposes
        roll_correction = np.sin(angles) * roll_correction  # Example correction based on angle 
        # Apply the correction to the ranges
        laser_scans += roll_correction  # Apply the correction to the ranges
        # for i in range(len(laser_scans)):
        #     # Apply correction to each range value
        #     laser_scans[i] += roll_correction
        t2 = rospy.Time.now()

        print(f"Processed scan in {(t2 - t1).to_sec()} seconds")

if __name__ == '__main__':

    lidar_correction = LidarCorrection()
    try:
        rospy.spin()  # Keep the node running
    except rospy.ROSInterruptException:
        pass  # Handle shutdown gracefully
        rospy.loginfo("Lidar correction node shutting down.")
    

