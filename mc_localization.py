import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, MapMetaData
from visualization_msgs.msg import MarkerArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker
import tf
from tf2_msgs.msg import TFMessage

import numpy as np
import matplotlib.pyplot as plt
from threading import Thread, Lock
from utils.grids import StochOccupancyGrid2D, DetOccupancyGrid2D
import time

SQRT6DIV2 = np.sqrt(6)/2
LIDAR_MAX_RANGE = 16  # FIXME TO BE DETERMINED

class MonteCarloLocalization:
    """
    The Monte Carlo Localization node

    Attributes:
        alpha1: The alpha1 parameter for the motion model
        alpha2: The alpha2 parameter for the motion model
        alpha3: The alpha3 parameter for the motion model
        alpha4: The alpha4 parameter for the motion model
        z_hit: The z_hit parameter for the measurement model
        sigma_hit: The sigma_hit parameter for the measurement model
        z_random: The z_random parameter for the measurement model
        pose: The estimated pose of the robot
        odom: The odometry data
        prev_odom: The previous odometry data
        num_particles: The number of particles to use in the particle filter
        prev_particles: The previous particles
        particles: The current particles
        odom_sub: The subscriber to the /odom topic
        scan_sub: The subscriber to the /scan topic
        map_sub: The subscriber to the /map topic
        map_md_sub: The subscriber to the /map_metadata topic
        pose_pub: The publisher to the /pose topic
        particle_pub: The publisher to the /particles topic
        have_map: A flag indicating whether the map has been received
        occupancy: The occupancy grid
        map_width: The width of the map
        map_height: The height of the map
        map_resolution: The resolution of the map
        map_origin: The origin of the map
        dist_lookup_table: A table of distances from the LIDAR sensor to the nearest occupied cell
        z_hit: The z_hit parameter for the measurement model
        sigma_hit: The sigma_hit parameter for the measurement model
        z_random: The z_random parameter for the measurement model
        mutex: A mutex for thread safety

    Description:
        Implements Monte Carlo localization using a particle filter. Measurement updates are based on results from a
        LIDAR sensor, and motion updates are based on odometry data. The node subscribes to the /odom and /scan topics
        and publishes the estimated pose to the /pose topic. The particles are published to the /particles topic for
        visualization in rviz.
    """

    def __init__(self, num_particles=300, alpha1=0.05, alpha2=0.05, alpha3=0.01, alpha4=0.001, sigma_hit=0.3, z_hit=0.75, z_random=0.25):
        """
        Initializes the Monte Carlo Localization node
        """
        # TODO, eventually change function inputs to be ROS parameters

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4

        rospy.init_node('monte_carlo_localization')

        self.pose = np.array([0, 0, 0])
        self.odom = None
        self.prev_odom = None
        self.moving = False

        self.num_particles = num_particles
        self.prev_particles = None
        self.particles = None
        self.pub_particle_indx = 0

        self.mutex = Lock()

        self.tf_listener = tf.TransformListener()

        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.map_md_sub = rospy.Subscriber('/map_metadata', MapMetaData, self.map_md_callback)

        self.tf_pub = rospy.Publisher("/tf", TFMessage, queue_size=10)
        self.particle_pub = rospy.Publisher('/particles', MarkerArray, queue_size=10)

        self.have_map = False
        self.occupancy = None
        self.map_width = None
        self.map_height = None
        self.map_resolution = None
        self.map_origin = None

        # The parameters for the mixing model of probabilities. z_hit + z_random = 1. These weight the probabilities
        self.z_hit = z_hit
        self.sigma_hit = sigma_hit  # standard deviation of the Gaussian
        self.z_random = z_random
        assert self.z_hit + self.z_random == 1, "z_hit + z_random must equal 1"

        # This table will be the lookup table for the distance to the nearest obstacle
        self.dist_lookup_table = np.load('lookup_table/mattbot_map.npy')
        unknown_indx = np.where(self.dist_lookup_table == -1)

        # Generate the probability lookup table based on distance to nearest obstacle
        self.prob_lookup_table = self.z_hit / np.sqrt(2 * np.pi * (self.sigma_hit ** 2)) * np.exp(
            -0.5 * (self.dist_lookup_table) ** 2 / (self.sigma_hit ** 2)) + self.z_random / LIDAR_MAX_RANGE
        self.prob_lookup_table[unknown_indx] = 1 / LIDAR_MAX_RANGE
        
        # fig, ax = plt.subplots()
        # cbar = ax.imshow(self.prob_lookup_table, cmap='hot')
        # fig.colorbar(cbar)
        # plt.show()


    def map_md_callback(self, msg):
        """
        Callback function for the map metadata subscriber

        Receives map metadata and stores it

        Args:
            msg: MapMetaData message
        """
        if (self.have_map is False):
            # Store map metadata
            self.map_width = msg.width
            self.map_height = msg.height
            self.map_resolution = msg.resolution
            self.map_origin = (msg.origin.position.x, msg.origin.position.y)

    def map_callback(self, msg):
        """
        Callback function for the map subscriber

        Receives new map info and updates our internal map representation

        Args:
            msg: OccupancyGrid message
        """

        # If we've received the map metadata and have a way to update it:
        if (self.have_map is False and self.map_width is not None and self.map_height is not None):

            # Create occupancy grid object
            self.occupancy = StochOccupancyGrid2D(
                self.map_resolution,
                self.map_width,
                self.map_height,
                self.map_origin[0],
                self.map_origin[1],
                5,
                msg.data,
            )

            # Initialize the particles
            print("Initializing particles...")
            self.init_particles()
            print("Particles initialized")

            self.have_map = True

    def odom_callback(self, msg):
        """
        Callback function for the odometry subscribe

        Updates the particles based on the odometry data by colling sample_motion_model_with_map

        Args:
            msg: Odometry message
        """
        
        self.mutex.acquire()
        t1 = time.time()


        # Get the x, y, and theta from the odometry message
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        orientation = msg.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, theta = euler_from_quaternion(orientation_list)

        # Sometimes odometry data is bad and we get really large values, only update if have reasonable values
        if np.abs(x) < 1e5 and np.abs(y) < 1e5 and np.abs(theta) < 1e5:

            self.odom = np.array([x, y, theta])
        
            if self.prev_odom is not None:

                # Only update particles if the odometry has changed
                if not np.array_equal(self.prev_odom, self.odom):
                    self.moving = True

                    # Update the particles based on the odometry data
                    u = np.array([self.prev_odom, self.odom]).T
                    if self.particles is not None:

                        # Loop through each particle and update it based on the odometry data
                        # for i in range(self.num_particles):
                        #     self.particles[:, i] = self.sample_motion_model_with_map(u, self.particles[:, i])
                        self.particles = self.sample_motion_model_with_map1(u, self.particles)
                        print("Motion Model Update")

                    # Publish the particles for visualization
                    if self.pub_particle_indx == 30:
                        self.pub_particle_indx = 0
                        self.publish_particles()
                    self.pub_particle_indx += 1
                    # self.publish_particles()
                    
                    t2 = time.time()
                    print("Motion: ", t2-t1)
                else:
                    self.moving = False

            # Store odometry data for next time
            self.prev_odom = self.odom

        
        self.mutex.release()
        

    def scan_callback(self, msg):
        """
        The callback for the laser scan subscriber

        Args:
            msg: LaserScan message
        """
        
        self.mutex.acquire()
        t1 = time.time()


        # Get the LIDAR measurements
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment
        range_min = msg.range_min
        range_max = msg.range_max

        angles = np.arange(angle_min, angle_max, angle_increment)
        
        ranges = ranges[::2]
        angles = angles[::2]
        valid_indx = np.where((ranges < range_max) & (ranges > range_min))

        # delete out of range values
        # ranges = np.delete(ranges, indx_max)
        # angles = np.delete(angles, indx_max)
        # ranges = np.delete(ranges, indx_min)
        # angles = np.delete(angles, indx_min)

        ranges = ranges[valid_indx]
        angles = angles[valid_indx]

        # delete nan values
        # nan_indx = np.where(np.isnan(ranges))
        # ranges = np.delete(ranges, nan_indx)
        # angles = np.delete(angles, nan_indx)

        if self.have_map and self.moving:
            # w = np.zeros(self.num_particles)
            # for i in range(self.num_particles):
            #     # w[i] = self.measurement_model_loop(ranges, self.particles[:, i], angles)
            #     w[i] = self.measurement_model1(ranges, self.particles[:, i], angles)

            t3 = time.time()
            # Get particle weights based on measurements
            w = self.measurement_model2(ranges, self.particles, angles)
            # print("Measurement Model Update")
            t4 = time.time()
            # Resample the particles based on the weights
            self.resample(w)

            # Publish the particles for visualization
            # self.publish_particles()

            t2 = time.time()
            print("Measurement Total: ", t2-t1)
            # print("Measurement Model: ", t4-t3)
            # print("Measurement Resample: ", t2-t4)
        self.mutex.release()

    def init_particles(self):
        """
        Initializes the particles distributed uniformly across the map
        """
        self.particles = np.zeros((3, self.num_particles))
        x = np.random.uniform(0, self.map_width*self.map_resolution-self.map_resolution, self.num_particles)
        y = np.random.uniform(0, self.map_height*self.map_resolution-self.map_resolution, self.num_particles)
        theta = np.random.uniform(-np.pi, np.pi, self.num_particles)
        for i in range(self.num_particles):
            while not self.occupancy.is_free((x[i], y[i])) or self.occupancy.is_unknown((x[i], y[i])):
                x[i] = np.random.uniform(0, self.map_width*self.map_resolution-self.map_resolution)
                y[i] = np.random.uniform(0, self.map_height*self.map_resolution-self.map_resolution)
            self.particles[0, i] = x[i]
            self.particles[1, i] = y[i]
            self.particles[2, i] = theta[i]
        self.publish_particles()

    def sample_motion_model_with_map(self, u, x_prev):
        """
        Samples the motion model with the map
        """
        pi = 0
        num_stuck = 0
        while pi <= 0:
            x = self.sample_motion_model(u, x_prev)
            pi = self.occupancy.is_free(x)
            # pi = self.occupancy.prob_x_given_map(x)
            num_stuck += 1
            
            if num_stuck > 10:
                return x_prev
        return x

    def sample_motion_model_with_map1(self, u, x_prev):
        """
        Samples the motion model with the map, does not require loop in M

        Args:
            x_prev: The previous pose of the robot, a 3xM array (x, y, theta)
        """
        pi = 1
        num_stuck = 0
        not_free_indx = np.arange(x_prev.shape[1])
        x = np.copy(x_prev)
        while len(not_free_indx) > 0:
            if num_stuck > 10:
                x[: , not_free_indx] = x_prev[: , not_free_indx]
                return x

            x[:, not_free_indx] = self.sample_motion_model1(u, x_prev[:, not_free_indx])
            pi = self.occupancy.is_free1(x)
            not_free_indx = np.where(pi == False)[0]
            num_stuck += 1
        return x

    def sample_motion_model(self, u, x_prev):
        """
        Samples the motion model based on odometry control. See Probabilistic Robotics, Table 5.6 pg 136

        Args:
            u: The control via odometry, a 3x2 array where the first column is the (x,y,theta) of previous time step from
                odometry and the second column is the (x,y,theta) of the current time step from odometry
            x_prev: The previous pose of the robot, a 3x1 array (x, y, theta)
        """
        delta_rot1 = np.arctan2(u[1, 1] - u[1, 0], u[0, 1] - u[0, 0]) - u[2, 0]
        delta_trans = np.sqrt((u[0, 1] - u[0, 0])**2 + (u[1, 1] - u[1, 0])**2)
        delta_rot2 = u[2, 1] - u[2, 0] - delta_rot1

        delta_rot1_hat = delta_rot1 - self.sample_normal(self.alpha1*np.abs(delta_rot1) + self.alpha2*delta_trans)
        delta_trans_hat = delta_trans - self.sample_normal(self.alpha3*delta_trans + self.alpha4*(np.abs(delta_rot1) + np.abs(delta_rot2)))
        delta_rot2_hat = delta_rot2 - self.sample_normal(self.alpha1*np.abs(delta_rot2) + self.alpha2*delta_trans)

        xp = x_prev[0] + delta_trans_hat * np.cos(x_prev[2] + delta_rot1_hat)
        yp = x_prev[1] + delta_trans_hat * np.sin(x_prev[2] + delta_rot1_hat)
        tp = x_prev[2] + delta_rot1_hat + delta_rot2_hat
        tp = self.wrap_theta(tp)

        return np.array([xp, yp, tp])

    def sample_motion_model1(self, u, x_prev):
        """
        Samples the motion model based on odometry control. See Probabilistic Robotics, Table 5.6 pg 136

        Args:
            u: The control via odometry, a 3x2 array where the first column is the (x,y,theta) of previous time step from
                odometry and the second column is the (x,y,theta) of the current time step from odometry
            x_prev: The previous pose of the robot, a 3x1 array (x, y, theta)
        """
        M = x_prev.shape[1]

        delta_rot1 = np.arctan2(u[1, 1] - u[1, 0], u[0, 1] - u[0, 0]) - u[2, 0]
        delta_trans = np.sqrt((u[0, 1] - u[0, 0]) ** 2 + (u[1, 1] - u[1, 0]) ** 2)
        delta_rot2 = u[2, 1] - u[2, 0] - delta_rot1

        delta_rot1_hat = delta_rot1 - self.sample_normal(self.alpha1 * np.abs(delta_rot1) + self.alpha2 * delta_trans, m=M)
        delta_trans_hat = delta_trans - self.sample_normal(
            self.alpha3 * delta_trans + self.alpha4 * (np.abs(delta_rot1) + np.abs(delta_rot2)), m=M)
        delta_rot2_hat = delta_rot2 - self.sample_normal(self.alpha1 * np.abs(delta_rot2) + self.alpha2 * delta_trans, m=M)

        xp = x_prev[0, :] + delta_trans_hat * np.cos(x_prev[2, :] + delta_rot1_hat)
        yp = x_prev[1, :] + delta_trans_hat * np.sin(x_prev[2, :] + delta_rot1_hat)
        tp = x_prev[2, :] + delta_rot1_hat + delta_rot2_hat
        tp = self.wrap_theta(tp)

        return np.vstack((xp, yp, tp))

    def sample_normal(self, b, m=None):
        """
        Samples a value from a normal distribution with mean 0 and standard deviation b
        """
        return np.random.normal(0, b, size=m)

    def sample_triangular(self, b):
        """
        Samples a value from a triangular distribution with mean 0 and standard deviation b
        """
        return SQRT6DIV2 * (np.random.uniform(-b, b) + np.random.uniform(-b, b))
        
    def wrap_theta(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def measurement_model0(self, z, x, theta_sens):
        """
        The measurement model for the LIDAR sensor. This is a likelihood model using distance to nearest neighbor
        See Probabilistic Robotics, Table 6.3 pg 172

        This model achieves the measurement model with a loop in z (measurement) and in x (particles)
        This implementation is slow...

        Args:
            z: The LIDAR measurement, a 1xN array where N is the number of measurements, we assume all measurements
                outside of the max range are already removed from this set
            x: The pose of the robot, a 3x1 array (x, y, theta)
            theta_sens: The angle of the sensor relative to the robot's frame, a 1xN array
        """
        q = 1
        for i in range(len(z)):
            x_meas = x[0] + z[i] * np.cos(x[2] + theta_sens[i])
            y_meas = x[1] + z[i] * np.sin(x[2] + theta_sens[i])

            x_grid = np.round(x_meas / self.map_resolution).astype(int)
            y_grid = np.round(y_meas / self.map_resolution).astype(int)

            if x_grid < 0 or x_grid >= self.map_width or y_grid < 0 or y_grid >= self.map_height:
                p = 1/LIDAR_MAX_RANGE
            else:
                p = self.prob_lookup_table[y_grid, x_grid]
            q *= p
        return q

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
        """
        n = len(z)  # number of measurements

        
        # Tile the x array to match the number of measurements
        x_tiled = np.tile(x[:, :, np.newaxis], (1, 1, n))

        # t1 = time.time()
        # Calculate the x and y coordinates of the measurements in the map frame
        x_meas = x_tiled[0, :, :] + z * np.cos(x_tiled[2, :, :] + theta_sens)
        y_meas = x_tiled[1, :, :] + z * np.sin(x_tiled[2, :, :] + theta_sens)
        # t2 = time.time()
        # print(t2 - t1)        

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

    def resample(self, w):
        """
        Resamples the particles: selects a new set of particles based on the weights

        Args:
            X: The particles: a 3xN array, where N = num_particles
            w: The weights: a 1xN array, where N = num_particles, and the sum of the weights is 1
        """

        # normalize the weights to get a probability distribution
        w = w / np.sum(w)

        # Resample the particles
        resample_indx = np.random.choice(np.arange(self.num_particles), size=self.num_particles, replace=True, p=w)

        # Update the particles
        self.particles = self.particles[:, resample_indx]

    def estimate_pose(self):
        """
        Estimates the robot's pose based on the particles and their weights
        """
        return np.mean(self.particles, axis=1)

    def publish_map_odom_transform(self, mo_x, mo_y, mo_theta):
        """
        Publishes a transform between the map and odom frames
        """
        # th1 = np.arctan2(self.pose[1], self.pose[0])

        th2 = np.arctan2(mo_y, mo_x)
        hyp = np.sqrt(mo_x**2 + mo_y**2)

        new_x = hyp * np.cos(self.pose[2] + th2)
        new_y = hyp * np.sin(self.pose[2] + th2)
        new_th = self.pose[2] + mo_theta

        # Create transform message
        tf_msg = TFMessage()

        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "map"
        transform.child_frame_id = "odom"
        transform.transform.translation.x = new_x
        transform.transform.translation.y = new_y
        transform.transform.translation.z = 0.0

        quat = quaternion_from_euler(0, 0, new_th)
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]

        # Publish transform
        tf_msg.transforms.append(transform)
        self.tf_pub.publish(tf_msg)

    def publish_particles(self):
        """
        Publishes the particles as a MarkerArray for visualization in rviz
        """
        particle_msg = MarkerArray()
        for i in range(self.num_particles):
            marker = Marker()
            marker.id = i
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = 'map'
            marker.pose.position.x = self.particles[0, i]
            marker.pose.position.y = self.particles[1, i]
            marker.pose.orientation.w = 1
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.025
            marker.scale.y = 0.025
            marker.scale.z = 0.025
            marker.color.a = 1.0  # Not transparent
            marker.color.r = 1.0  # Red
            marker.color.g = 0.0  # Green
            marker.color.b = 0.0  # Blue
            particle_msg.markers.append(marker)
        self.particle_pub.publish(particle_msg)

    def run(self):
        """
        Runs the node
        """
        # Run the node, every 0.5 seconds estimate the pose and publish it
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            if self.have_map:
                self.pose = self.estimate_pose()

                try:
                    (trans, rot) = self.tf_listener.lookupTransform("/odom", "/base_footprint", rospy.Time(0))
                    x = trans[0]
                    y = trans[1]
                    _, _, theta = euler_from_quaternion(rot)
                    print("Transform from odom to base_footprint:")
                    print("x, y, theta: ", x, y, theta)

                    self.publish_map_odom_transform(x, y, theta)
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    print("Failed to lookup transform from odom to base_footprint")
            rate.sleep()

    def shutdown(self):
        """
        Shutdown function
        """
        rospy.loginfo("Shutting down monte_carlo_localization node...")
        rospy.sleep(1)

if __name__ == '__main__':
    mcl = MonteCarloLocalization()
    rospy.on_shutdown(mcl.shutdown)
    mcl.run()
