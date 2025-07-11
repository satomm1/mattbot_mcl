import rospy
import rospkg
import tf
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose2D, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import MapMetaData

import numpy as np

LIDAR_MAX_RANGE = 16

class LocalizationQuality:
    def __init__(self):
        # Initialize the node
        rospy.init_node('localization_quality', anonymous=True)

        # This table will be the lookup table for the distance to the nearest obstacle
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('mattbot_mcl')
        data_path = pkg_path + '/lookup_table/current_map.npy'
        self.dist_lookup_table = np.load(data_path)
        unknown_indx = np.where(self.dist_lookup_table == -1)

        self.z_hit = 0.75
        self.z_random = 0.25
        self.sigma_hit = 0.1  # Standard deviation for the Gaussian distribution

        self.lidar_measurement_skip = 2  # Skip every nth measurement to speed up computations

        # Generate the probability lookup table based on distance to nearest obstacle
        self.prob_lookup_table = self.z_hit / np.sqrt(2 * np.pi * (self.sigma_hit ** 2)) * np.exp(
            -0.5 * (self.dist_lookup_table) ** 2 / (self.sigma_hit ** 2)) + self.z_random / LIDAR_MAX_RANGE
        self.prob_lookup_table[unknown_indx] = 1 / LIDAR_MAX_RANGE

        # Wait for /map_metadata message
        map_md = rospy.wait_for_message('/map_metadata', MapMetaData)
        self.map_resolution = map_md.resolution
        self.map_width = map_md.width
        self.map_height = map_md.height

        self.localized = False
        self.match_with_map = False

        # Store the last 30 locations and weights so that we can determine if we have lost localization
        self.max_history_length = 30
        self.location_history = [] 
        self.weight_history = []

        # Subscribe to laser scan
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        self.localized_sub = rospy.Subscriber('/localized', Bool, self.localized_callback, queue_size=10)
        
        self.lost_localization_pub = rospy.Publisher('/lost_localization', Bool, queue_size=10)
        self.initial_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)

        # Transform listener to get the laser frame
        self.trans_listener = tf.TransformListener()

    def measurement_model1(self, z, x, theta_sens):
        """
        The measurement model for the LIDAR sensor. This is a likelihood model using distance to nearest neighbor
        See Probabilistic Robotics, Table 6.3 pg 172

        This model achieves the measurement model with no loop in z. But, still requires only a single x input.
        This model achieves approx 10x speedup over measurement_model0, which has to loop over measurements

        Args:
            z: The LIDAR measurement, a 1xN array where N is the number of measurements, we assume all measurements
                outside of the max range are already removed from this set
            x: The pose of the robot, a 3x1 array (x, y, theta)
            theta_sens: The angle of the sensor relative to the robot's frame, a 1xN array

        Returns:
            The likelihood of the measurement given the pose
        """
        # Calculate the x and y coordinates of the measurements in the map frame
        x_meas = x[0] + z*np.cos(x[2] + theta_sens)
        y_meas = x[1] + z*np.sin(x[2] + theta_sens)

        # convert x_meas and y_meas to grid coordinates
        x_grid = np.round(x_meas/self.map_resolution).astype(int)
        y_grid = np.round(y_meas/self.map_resolution).astype(int)

        # Get indices of out of range locations
        out_of_range_x = np.where((x_grid < 0) | (x_grid >= self.map_width))
        out_of_range_y = np.where((y_grid < 0) | (y_grid >= self.map_height))

        # Clip the grid coordinates to be within the map
        x_grid_norm = np.clip(x_grid, 0, self.map_width-1)
        y_grid_norm = np.clip(y_grid, 0, self.map_height-1)

        # Look up the probabilities from the precomputed table
        p = self.prob_lookup_table[y_grid_norm, x_grid_norm]

        # Set out of range locations to 1 / LIDAR_MAX_RANGE (these are unknown locations)
        p[out_of_range_x] = 1/LIDAR_MAX_RANGE
        p[out_of_range_y] = 1/LIDAR_MAX_RANGE

        # Instead of doing product of all probabilities, we sum p^3 as a heuristic
        return np.sum(np.power(p,3))

    def measurement_model2(self, z, x, theta_sens):
        """
        The measurement model for the LIDAR sensor. This is a likelihood model using distance to nearest neighbor
        See Probabilistic Robotics, Table 6.3 pg 172

        This model achieves the measurement model with no loop in z and takes multiple x particles as input
        This model achieves approx 10x speedup over measurement_model1, which has to loop over particles

        Args:
            z: The LIDAR measurement, a 1xN array where N is the number of measurements, we assume all measurements
                outside of the max range are already removed from this set
            x: The pose of the robot, a 3xM array (x, y, theta), M is number of particles
            theta_sens: The angle of the sensor relative to the robot's frame, a 1xN array

        Returns:
            The likelihood of the measurement given the poses, a 1xM array
        """
        n = len(z)  # number of measurements

        # Tile the x array to match the number of measurements
        x_tiled = np.tile(x[:, :, np.newaxis], (1, 1, n))

        # Calculate the x and y coordinates of the measurements in the map frame
        x_meas = x_tiled[0, :, :] + z * np.cos(x_tiled[2, :, :] + theta_sens)
        y_meas = x_tiled[1, :, :] + z * np.sin(x_tiled[2, :, :] + theta_sens)

        # convert x_meas and y_meas to grid coordinates
        x_grid = np.round(x_meas / self.map_resolution).astype(int)
        y_grid = np.round(y_meas / self.map_resolution).astype(int)

        # Get indices of out of range locations
        out_of_range_x = np.where((x_grid < 0) | (x_grid >= self.map_width))
        out_of_range_y = np.where((y_grid < 0) | (y_grid >= self.map_height))

        # Clip the grid coordinates to be within the map
        x_grid_norm = np.clip(x_grid, 0, self.map_width - 1)
        y_grid_norm = np.clip(y_grid, 0, self.map_height - 1)

        # Look up the probabilities from the precomputed table
        p = self.prob_lookup_table[y_grid_norm, x_grid_norm]

        # Set out of range locations to 1 / LIDAR_MAX_RANGE (these are unknown locations)
        p[out_of_range_x[0], out_of_range_x[1]] = 1 / LIDAR_MAX_RANGE
        p[out_of_range_y[0], out_of_range_y[1]] = 1 / LIDAR_MAX_RANGE

        # Instead of doing product of all probabilities, we sum p^3 as a heuristic
        return np.sum(np.power(p, 3), axis=1)

    def scan_callback(self, msg):
        """
        The callback for the laser scan subscriber. This function is called whenever a new laser scan message is
        received. The function updates the particles based on the laser scan data using the measurement model. The
        particles are then resampled based on the weights calculated from the measurement model.

        Args:
            msg: LaserScan message
        """

        # Get the LIDAR measurements from the message
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment
        range_min = msg.range_min
        range_max = 3

        # Get corresponding angles based on the message data
        angles = np.arange(angle_min, angle_max, angle_increment)

        # Don't use every measurement to speed up computations
        ranges = ranges[::self.lidar_measurement_skip]
        angles = angles[::self.lidar_measurement_skip]

        # Only use measurements that are within valid range
        valid_indx = np.where((ranges < range_max) & (ranges > range_min))
        if len(valid_indx[0]) < 200: 
            valid_indx = np.where((ranges < range_max+3) & (ranges > range_min))
        ranges = ranges[valid_indx]
        angles = angles[valid_indx]

        try: 
            (translation, rotation) = self.trans_listener.lookupTransform(
                        "/map", "/laser_frame", rospy.Time(0)
                    )
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        laser_x = translation[0]
        laser_y = translation[1]
        laser_theta = tf.transformations.euler_from_quaternion(rotation)[2]

        # Current Pose of the robot
        current_pose = np.array([[laser_x, laser_y, laser_theta]]).T

        w = self.measurement_model1(ranges, current_pose, angles)

        # Store the last 30 locations and weights
        self.location_history.append(current_pose)
        self.weight_history.append(w)
        if len(self.location_history) > self.max_history_length:
            self.location_history.pop(0)
            self.weight_history.pop(0)

        # Check if we have lost localization
        if True: #self.localized and self.match_with_map:
            if len(self.weight_history) < self.max_history_length:
                return
            
            # Check if the weights have dropped significantly-compare most recent 10 weights to the average of the last 10
            recent_weights = self.weight_history[-10:]
            old_weights = self.weight_history[10:-10]
            if np.mean(recent_weights) < 0.5 * np.mean(old_weights):
                # We lost localization
                rospy.loginfo("Lost localization")
                self.localized = False
                self.match_with_map = False
                self.lost_localization_pub.publish(Bool(data=True))

                # Get the pose that best matches the laser scans
                weights = self.measurement_model2(ranges, np.array(self.location_history).squeeze(), angles)
                best_index = np.argmax(weights)

                print("Best index:", best_index)

                # Now get the last good pose and publish it
                last_good_pose = self.location_history[best_index]
                last_good_pose_msg = PoseWithCovarianceStamped()
                last_good_pose_msg.header.stamp = rospy.Time.now()
                last_good_pose_msg.header.frame_id = "map"
                last_good_pose_msg.pose.pose.position.x = last_good_pose[0, 0]
                last_good_pose_msg.pose.pose.position.y = last_good_pose[1, 0]
                last_good_pose_msg.pose.pose.position.z = 0.0
                quat = tf.transformations.quaternion_from_euler(0.0, 0.0, last_good_pose[2, 0])
                last_good_pose_msg.pose.pose.orientation.x = quat[0]
                last_good_pose_msg.pose.pose.orientation.y = quat[1]
                last_good_pose_msg.pose.pose.orientation.z = quat[2]
                last_good_pose_msg.pose.pose.orientation.w = quat[3]
                covariance = np.zeros((6, 6))
                covariance[0, 0] = 0.1  # x variance
                covariance[1, 1] = 0.1  # y variance
                covariance[5, 5] = 3.14  # theta variance
                last_good_pose_msg.pose.covariance = covariance.flatten().tolist()

                # self.initial_pose_pub.publish(last_good_pose_msg)

        # print(rospy.get_time())
        

    def localized_callback(self, msg):
        if not self.localized and msg.data:
            rospy.loginfo("Robot is localized")
            self.localized = True
            self.match_with_map = True

    def run(self):
        """
        The main loop of the node. This function keeps the node running and processing laser scan data.
        """
        rospy.spin()

if __name__ == '__main__':
    try:
        localization_quality = LocalizationQuality()
        localization_quality.run()
    except rospy.ROSInterruptException:
        pass