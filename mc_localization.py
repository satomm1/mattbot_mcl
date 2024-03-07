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

    def __init__(self, num_particles=300, alpha1=0.05, alpha2=0.05, alpha3=0.01, alpha4=0.001, sigma_hit=0.1, z_hit=0.75, z_random=0.25, lidar_measurement_skip=2, visualize=False):
        """
        Initializes the Monte Carlo Localization node
        """
        # TODO, eventually change function inputs to be ROS parameters

        # Parameters for the motion model
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4

        # Initialize the node
        rospy.init_node('monte_carlo_localization')

        # Initialize the pose and other variables needed
        self.pose = np.array([0, 0, 0])
        self.odom = None
        self.prev_odom = None
        self.moving = False
        self.motion_updated = False
        self.updates_after_stopping = 0 # Number of measurment updates after the robot has stopped

        # Initialize the particles
        self.num_particles = num_particles
        self.prev_particles = None
        self.particles = None
        self.pub_particle_indx = 0

        # Initialize the skip for the LIDAR measurements (i.e. 1 means we use every measurement, 2 means every other, etc.)
        self.lidar_measurement_skip = lidar_measurement_skip

        # Initialize the mutex needed for thread safety
        self.mutex = Lock()

        # Create a transform listener
        self.tf_listener = tf.TransformListener()

        # Create the subscribers
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.map_md_sub = rospy.Subscriber('/map_metadata', MapMetaData, self.map_md_callback)

        # Create the publishers
        self.tf_pub = rospy.Publisher("/tf", TFMessage, queue_size=10)
        self.particle_pub = rospy.Publisher('/particles', MarkerArray, queue_size=10)

        # Initialize the map variables
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

        # Plot the probability lookup table if requested
        if visualize:
            fig, ax = plt.subplots()
            cbar = ax.imshow(self.prob_lookup_table, cmap='hot')
            fig.colorbar(cbar)
            plt.show()

    def map_md_callback(self, msg):
        """
        Callback function for the map metadata subscriber

        Receives map metadata and stores it. Only stores once so if we have a map update we won't process it

        Args:
            msg: MapMetaData message
        """
        if self.have_map is False:
            # Store map metadata
            self.map_width = msg.width
            self.map_height = msg.height
            self.map_resolution = msg.resolution
            self.map_origin = (msg.origin.position.x, msg.origin.position.y)

    def map_callback(self, msg):
        """
        Callback function for the map subscriber

        Receives new map info and updates our internal map representation using the StochOccupancyGrid2D class
        Only updates once so if we have a map update we won't process it
        After getting the map, the particles are initialized

        Args:
            msg: OccupancyGrid message
        """

        # If we've received the map metadata and have a way to update it:
        if self.have_map is False and self.map_width is not None and self.map_height is not None:

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
        Callback function for the odometry subscriber

        Gets the x,y,theta updates from odometry data. If change in odometry is nonzero, calls the motion model to
        update the particles based on the odometry data by calling sample_motion_model_with_map. Publishes the
        updated particles periodically.

        Args:
            msg: Odometry message
        """

        # Acquire the thread lock
        self.mutex.acquire()

        # Get the x, y, and theta from the odometry message
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, theta = euler_from_quaternion(orientation_list)

        # Sometimes odometry data is bad and has really large values, only update if the values are reasonable
        if np.abs(x) < 1e5 and np.abs(y) < 1e5 and np.abs(theta) < 1e5:

            # Store the odometry data
            self.odom = np.array([x, y, theta])

            # Only do motion model update if we have a previous odometry measurement
            if self.prev_odom is not None:

                # Only update particles if the odometry has changed (i.e. the robot has moved)
                if not np.array_equal(self.prev_odom, self.odom):
                    self.moving = True
                    self.motion_updated = True

                    # Update the particles based on the odometry data
                    u = np.array([self.prev_odom, self.odom]).T  # The control array
                    if self.particles is not None:
                        self.particles = self.sample_motion_model_with_map1(u, self.particles)  # Motion model update

                    # Publish the particles for visualization (only update periodically)
                    if self.pub_particle_indx == 30:
                        self.pub_particle_indx = 0
                        self.publish_particles()
                    self.pub_particle_indx += 1

                else:
                    self.moving = False  # The robot hasn't moved

            # Store odometry data for next time
            self.prev_odom = self.odom

        # Release the thread lock
        self.mutex.release()

    def scan_callback(self, msg):
        """
        The callback for the laser scan subscriber. This function is called whenever a new laser scan message is
        received. The function updates the particles based on the laser scan data using the measurement model. The
        particles are then resampled based on the weights calculated from the measurement model.

        Args:
            msg: LaserScan message
        """

        # Acquire the thread lock
        self.mutex.acquire()

        # Get the LIDAR measurements from the message
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment
        range_min = msg.range_min
        range_max = msg.range_max

        # Get corresponding angles based on the message data
        angles = np.arange(angle_min, angle_max, angle_increment)

        # Don't use every measurement to speed up computations
        ranges = ranges[::self.lidar_measurement_skip]
        angles = angles[::self.lidar_measurement_skip]

        # Only use measurements that are within valid range
        valid_indx = np.where((ranges < range_max) & (ranges > range_min))
        ranges = ranges[valid_indx]
        angles = angles[valid_indx]

        # Only do measurement model update if there is a map and the robot is moving
        if self.have_map and (self.moving or self.updates_after_stopping < 10) and self.motion_updated:
            # Get particle weights based on measurements
            w = self.measurement_model2(ranges, self.particles, angles)

            # Resample the particles based on the weights
            self.resample(w)

            if not self.moving:
                self.updates_after_stopping += 1
            else:
                self.updates_after_stopping = 0

        # Release the thread lock
        self.mutex.release()

    def init_particles(self):
        """
        Initializes the particles distributed uniformly across the free space in the map
        """

        # Initialize the particles uniformly across map space
        self.particles = np.zeros((3, self.num_particles))

        # Get first attempt at particles without considering where they are
        x = np.random.uniform(0, self.map_width*self.map_resolution-self.map_resolution, self.num_particles)
        y = np.random.uniform(0, self.map_height*self.map_resolution-self.map_resolution, self.num_particles)
        theta = np.random.uniform(-np.pi, np.pi, self.num_particles)

        # Now ensure the particles are in free space
        for i in range(self.num_particles):
            # Do not allow in unknown locations or occupied locations
            while not self.occupancy.is_free((x[i], y[i])) or self.occupancy.is_unknown((x[i], y[i])):
                x[i] = np.random.uniform(0, self.map_width*self.map_resolution-self.map_resolution)
                y[i] = np.random.uniform(0, self.map_height*self.map_resolution-self.map_resolution)

            # Store the particle
            self.particles[0, i] = x[i]
            self.particles[1, i] = y[i]
            self.particles[2, i] = theta[i]

        # Publish the particles for visualization
        self.publish_particles()

    def sample_motion_model_with_map(self, u, x_prev):
        """
        This is the motion model when we have a map. It samples the motion model repeatedly until the new pose is in
        free space. This function uses self.sample_motion_model, which is a slow implementation that requires a loop
        over the number of particles.

        Args:
            u: The control via odometry, a 3x2 array where the first column is the (x,y,theta) of previous time step
                from odometry and the second column is the (x,y,theta) of the current time step from odometry
            x_prev: The previous pose of the robot, a 3xM array (x, y, theta)
        """
        pi = 0
        num_stuck = 0
        while pi <= 0:
            # Sample the motion model
            x = self.sample_motion_model(u, x_prev)

            # Check if the new pose is in free space
            pi = self.occupancy.is_free(x)

            # After ten tries, just return the previous pose
            num_stuck += 1
            if num_stuck > 10:
                return x_prev
        return x

    def sample_motion_model_with_map1(self, u, x_prev):
        """
        This is the motion model when we have a map. It samples the motion model repeatedly until the new pose is in
        free space. This function uses self.sample_motion_model1, which is a fast implementation that requires no loop
        over the number of particles.

        Args:
            u: The control via odometry, a 3x2 array where the first column is the (x,y,theta) of previous time step
                from odometry and the second column is the (x,y,theta) of the current time step from odometry
            x_prev: The previous pose of the robot, a 3xM array (x, y, theta)
        """
        num_stuck = 0
        not_free_indx = np.arange(x_prev.shape[1])
        x = np.copy(x_prev)

        # Sample the motion model repeatedly until the new pose is in free space
        while len(not_free_indx) > 0:
            # If we've tried too many times, just return the previous pose
            if num_stuck > 10:
                x[: , not_free_indx] = x_prev[: , not_free_indx]
                return x

            # Sample the motion model
            x[:, not_free_indx] = self.sample_motion_model1(u, x_prev[:, not_free_indx])

            # Check which new poses are in free space, and determine indices of those that are not
            pi = self.occupancy.is_free1(x)
            not_free_indx = np.where(pi == False)[0]
            num_stuck += 1
        return x

    def sample_motion_model(self, u, x_prev):
        """
        Samples the motion model based on odometry data. See Probabilistic Robotics, Table 5.6 pg 136

        Args:
            u: The control via odometry, a 3x2 array where the first column is the (x,y,theta) of previous time step from
                odometry and the second column is the (x,y,theta) of the current time step from odometry
            x_prev: The previous pose of the robot, a 3x1 array (x, y, theta)

        Returns:
            The new pose of the particles, a 3x1 array
        """

        # Implements the odometry sampling model, see Probabilistic Robotics for details
        delta_rot1 = np.arctan2(u[1, 1] - u[1, 0], u[0, 1] - u[0, 0]) - u[2, 0]
        delta_trans = np.sqrt((u[0, 1] - u[0, 0])**2 + (u[1, 1] - u[1, 0])**2)
        delta_rot2 = u[2, 1] - u[2, 0] - delta_rot1

        delta_rot1_hat = delta_rot1 - self.sample_normal(self.alpha1*np.abs(delta_rot1) + self.alpha2*delta_trans)
        delta_trans_hat = delta_trans - self.sample_normal(self.alpha3*delta_trans + self.alpha4*(np.abs(delta_rot1) + np.abs(delta_rot2)))
        delta_rot2_hat = delta_rot2 - self.sample_normal(self.alpha1*np.abs(delta_rot2) + self.alpha2*delta_trans)

        xp = x_prev[0] + delta_trans_hat * np.cos(x_prev[2] + delta_rot1_hat)
        yp = x_prev[1] + delta_trans_hat * np.sin(x_prev[2] + delta_rot1_hat)
        tp = x_prev[2] + delta_rot1_hat + delta_rot2_hat
        tp = self.wrap_theta(tp)  # Wrap the angle to be between -pi and pi

        return np.array([xp, yp, tp])

    def sample_motion_model1(self, u, x_prev):
        """
        Samples the motion model based on odometry control. See Probabilistic Robotics, Table 5.6 pg 136
        This implementation can handle multiple particles at once.

        Args:
            u: The control via odometry, a 3x2 array where the first column is the (x,y,theta) of previous time step from
                odometry and the second column is the (x,y,theta) of the current time step from odometry
            x_prev: The previous pose of the robot, a 3xM array, (x, y, theta), where M is the number of particles

        Returns:
            The new pose of the particles, a 3xM array
        """
        M = x_prev.shape[1]  # Number of particles

        # Implements the odometry sampling model, see Probabilistic Robotics for details
        delta_rot1 = np.arctan2(u[1, 1] - u[1, 0], u[0, 1] - u[0, 0]) - u[2, 0]
        delta_trans = np.sqrt((u[0, 1] - u[0, 0]) ** 2 + (u[1, 1] - u[1, 0]) ** 2)
        delta_rot2 = u[2, 1] - u[2, 0] - delta_rot1

        delta_rot1_hat = delta_rot1 - self.sample_normal(self.alpha1 * np.abs(delta_rot1) + self.alpha2 * delta_trans,
                                                         m=M)
        delta_trans_hat = delta_trans - self.sample_normal(
            self.alpha3 * delta_trans + self.alpha4 * (np.abs(delta_rot1) + np.abs(delta_rot2)), m=M)
        delta_rot2_hat = delta_rot2 - self.sample_normal(self.alpha1 * np.abs(delta_rot2) + self.alpha2 * delta_trans,
                                                         m=M)

        xp = x_prev[0, :] + delta_trans_hat * np.cos(x_prev[2, :] + delta_rot1_hat)
        yp = x_prev[1, :] + delta_trans_hat * np.sin(x_prev[2, :] + delta_rot1_hat)
        tp = x_prev[2, :] + delta_rot1_hat + delta_rot2_hat
        tp = self.wrap_theta(tp)

        return np.vstack((xp, yp, tp))

    def sample_normal(self, b, m=None):
        """
        Samples a value from a normal distribution with mean 0 and standard deviation b

        Args:
            b: The standard deviation of the normal distribution
            m: The number of samples to take

        Returns:
            A value from the normal distribution
        """
        return np.random.normal(0, b, size=m)

    def sample_triangular(self, b):
        """
        Samples a value from a triangular distribution with mean 0 and standard deviation b

        Args:
            b: The standard deviation of the triangular distribution

        Returns:
            A value from the triangular distribution
        """
        return SQRT6DIV2 * (np.random.uniform(-b, b) + np.random.uniform(-b, b))
        
    def wrap_theta(self, theta):
        """
        Wraps the equivalent angle in between -pi and pi

        Args:
            theta: The angle to be wrapped

        Returns:
            The equivalent angle in between -pi and pi
        """
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def measurement_model0(self, z, x, theta_sens):
        """
        The measurement model for the LIDAR sensor. This is a likelihood model using distance to nearest neighbor
        See Probabilistic Robotics, Table 6.3 pg 172.

        This model achieves the measurement model with a loop in z (measurement) and in x (particles)
        This implementation is slow.

        Args:
            z: The LIDAR measurement, a 1xN array where N is the number of measurements, we assume all measurements
                outside of the max range are already removed from this set
            x: The pose of the robot, a 3x1 array (x, y, theta)
            theta_sens: The angle of the sensor relative to the robot's frame, a 1xN array

        Returns:
            The likelihood of the measurement given the pose
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

    def resample(self, w):
        """
        Resamples the particles: selects a new set of particles based on the weights

        Args:
            w: The weights: a 1xN array, where N = num_particles
        """

        # normalize the weights to get a probability distribution
        w = w / np.sum(w)
        self.pose = self.estimate_pose(w)

        # Resample the particles using the weights
        resample_indx = np.random.choice(np.arange(self.num_particles), size=self.num_particles, replace=True, p=w)

        # Update the particles based on the resampling
        self.particles = self.particles[:, resample_indx]

    def estimate_pose(self, w):
        """
        Estimates the robot's pose based on the particles. Here, we take the weighted average of the particles as
        the estimated pose

        Args:
            w: The weights: a 1xN array, where N = num_particles

        Returns:
            The estimated pose of the robot, a 3x1 array (x, y, theta)
        """
        return np.average(self.particles, axis=1, weights=w)

    def publish_map_odom_transform(self, x_o_bf, y_o_bf, th_o_bf):
        """
        Publishes a transform between the map and odom frames. Calculates the map->odom transform based on the current
        estimated pose of the robot and the current odom->base_footprint transform. This transform is published as a
        TFMessage.

        Args:
            x_o_bf: The x position of the robot in the odom->base_footprint transform
            y_o_bf: The y position of the robot in the odom->base_footprint transform
            th_o_bf: The orientation of the robot in the odom->base_footprint transform
        """
        # Calculate the map->odom transform based on the current estimated pose of the robot and the current
        # odom->base_footprint transform, see figures/transform_geometry.png for details
        th2 = np.arctan2(y_o_bf, x_o_bf)
        h1 = np.sqrt(x_o_bf**2 + y_o_bf**2)

        th_m_o = self.pose[2] - th_o_bf

        x_m_o = self.pose[0] - h1 * np.cos(th2 + th_m_o)
        y_m_o = self.pose[1] - h1 * np.sin(th2 + th_m_o)

        # Create transform message
        tf_msg = TFMessage()

        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "map"
        transform.child_frame_id = "odom"
        transform.transform.translation.x = x_m_o
        transform.transform.translation.y = y_m_o
        transform.transform.translation.z = 0.0

        quat = quaternion_from_euler(0, 0, th_m_o)
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
        rate = rospy.Rate(2)  # 2 Hz
        while not rospy.is_shutdown():
            if self.have_map:
                # If estimated pose is outside of map boundaries, reset particle filter:
                if self.pose[0] < 0 or self.pose[0] > self.map_width*self.map_resolution or self.pose[1] < 0 or self.pose[1] > self.map_height*self.map_resolution:
                    self.mutex.acquire()
                    print("Resetting particles...")
                    self.init_particles()
                    print("Particles reset")
                    self.mutex.release()

                try:
                    # Get the transform from odom to base_footprint
                    (trans, rot) = self.tf_listener.lookupTransform("/odom", "/base_footprint", rospy.Time(0))
                    x = trans[0]
                    y = trans[1]
                    _, _, theta = euler_from_quaternion(rot)

                    # Publish the map->odom transform
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

    mcl = MonteCarloLocalization()  # Initialize the Monte Carlo Localization node
    rospy.on_shutdown(mcl.shutdown)  # Define the shutdown function
    mcl.run()  # Run the node
