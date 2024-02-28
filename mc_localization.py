import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, MapMetaData
from visualization_msgs.msg import MarkerArray
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker

import numpy as np
from threading import Thread, Lock
from utils.grids import StochOccupancyGrid2D, DetOccupancyGrid2D

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

    def __init__(self, num_particles=100, alpha1=1, alpha2=1, alpha3=1, alpha4=1, sigma_hit=0.1, z_hit=0.75, z_random=0.25):
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

        self.num_particles = num_particles
        self.prev_particles = None
        self.particles = None

        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.map_md_sub = rospy.Subscriber('/map_metadata', MapMetaData, self.map_md_callback)

        self.pose_pub = rospy.Publisher('/pose', PoseStamped, queue_size=10)
        self.particle_pub = rospy.Publisher('/particles', MarkerArray, queue_size=10)

        self.have_map = False
        self.occupancy = None
        self.map_width = None
        self.map_height = None
        self.map_resolution = None
        self.map_origin = None

        # This table will be the lookup table for the distance to the nearest obstacle
        self.dist_lookup_table = None

        # The parameters for the mixing model of probabilities. z_hit + z_random = 1. These weight the probabilities
        self.z_hit = z_hit
        self.sigma_hit = sigma_hit  # standard deviation of the Gaussian
        self.z_random = z_random
        assert self.z_hit + self.z_random == 1, "z_hit + z_random must equal 1"

        self.mutex = Lock()

    def map_md_callback(self, msg):
        """
        Callback function for the map metadata subscriber

        Receives map metadata and stores it

        Args:
            msg: MapMetaData message
        """
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
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:
        if (self.map_width > 0 and self.map_height > 0 and len(self.map_probs) > 0):

            self.occupancy = StochOccupancyGrid2D(
                self.map_resolution,
                self.map_width,
                self.map_height,
                self.map_origin[0],
                self.map_origin[1],
                5,
                self.map_probs,
            )

            if self.have_map is False:
                self.have_map = True
                self.generate_dist_lookup_table()

    def generate_dist_lookup_table(self):
        """
        Generates a table of distances from the LIDAR sensor to the nearest occupied cell
        """
        self.dist_lookup_table = np.zeros((self.map_width, self.map_height))
        for x in range(self.map_width):
            for y in range(self.map_height):
                if self.occupancy.is_free((x, y)):
                    self.dist_lookup_table[x, y] = self.find_closest_obstacle((x, y))
                else:
                    self.dist_lookup_table[x, y] = 0

    def find_closest_obstacle(self, x, y):
        """
        Finds the closest obstacle to a given cell

        Args:
            x: The x coordinate of the cell
            y: The y coordinate of the cell
        """
        k = 1
        while True:
            for i in np.arange(-k, k+1):
                for j in np.arange(-k, k+1):
                    if np.abs(i) == k or np.abs(j) == k:
                        new_x = np.clip(x+i, 0, self.map_width-1)
                        new_y = np.clip(y+j, 0, self.map_height-1)
                        if ~self.occupancy.is_free((new_x, new_y)):
                            return np.sqrt(i**2 + j**2)*self.map_resolution
            k += 1

    def odom_callback(self, msg):
        """
        Callback function for the odometry subscriber

        Args:
            msg: Odometry message
        """
        self.mutex.acquire()
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        orientation = msg.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, theta = euler_from_quaternion(orientation_list)

        self.mutex.release()

    def scan_callback(self, msg):
        """
        The callback for the laser scan subscriber

        Args:
            msg: LaserScan message
        """
        self.mutex.acquire()
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment
        range_min = msg.range_min
        range_max = msg.range_max

        angles = np.arange(angle_min, angle_max, angle_increment)
        indx_max = np.where(ranges > range_max)
        indx_min = np.where(ranges < range_min)

        # delete out of range values
        ranges = np.delete(ranges, indx_max)
        angles = np.delete(angles, indx_max)
        ranges = np.delete(ranges, indx_min)
        angles = np.delete(angles, indx_min)

        # delete nan values
        nan_indx = np.where(np.isnan(ranges))
        ranges = np.delete(ranges, nan_indx)
        angles = np.delete(angles, nan_indx)

        self.mutex.release()

    def sample_motion_model_with_map(self, u, x_prev):
        """
        Samples the motion model with the map
        """
        pi = 0
        while pi <= 0:
            x = self.sample_motion_model(u, x_prev)
            pi = self.occupancy.is_free(x)
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

        return np.array([xp, yp, tp])

    def sample_normal(self, b):
        """
        Samples a value from a normal distribution with mean 0 and standard deviation b
        """
        return np.random.normal(0, b)

    def sample_triangular(self, b):
        """
        Samples a value from a triangular distribution with mean 0 and standard deviation b
        """
        return SQRT6DIV2 * (np.random.uniform(-b, b) + np.random.uniform(-b, b))

    def measurement_model(self, z, x, theta_sens):
        """
        The measurement model for the LIDAR sensor. This is a likelihood model using distance to nearest neighbor
        See Probabilistic Robotics, Table 6.3 pg 172

        Args:
            z: The LIDAR measurement, a 1xN array where N is the number of measurements, we assume all measurements
                outside of the max range are already removed from this set
            x: The pose of the robot, a 3xN array (x, y, theta)
            theta_sens: The angle of the sensor relative to the robot's frame, a 1xN array
        """
        q = 1


        x_meas = x[0,:] + z*np.cos(x[2,:] + theta_sens)
        y_meas = x[1,:] + z*np.sin(x[2,:] + theta_sens)

        # convert x_meas and y_meas to grid coordinates
        x_grid = np.round(x_meas/self.map_resolution)
        y_grid = np.round(y_meas/self.map_resolution)

        dist = self.dist_lookup_table[x_grid, y_grid]
        p_hit = 1/np.sqrt(2*np.pi*(self.sigma_hit**2))*np.exp(-0.5*(dist)**2/(self.sigma_hit**2))
        p = self.z_hit*p_hit + self.z_random/LIDAR_MAX_RANGE

        q = q * p
        return q

    def resample(self, X, w):
        """
        Resamples the particles:

        Args:
            X: The particles: a 3xN array, where N = num_particles
            w: The weights: a 1xN array, where N = num_particles, and the sum of the weights is 1
        """
        return np.random.choice(X, size=self.num_particles, replace=True, p=w)

    def mcl(self, X_prev, u, z, theta_sens):
        """
        Implements the Monte Carlo Localization algorithm
        """
        X_bar = np.zeros((3, self.num_particles))
        w = np.zeros((1, self.num_particles))
        for m in range(self.num_particles):
            x_new = self.sample_motion_model_with_map(u, X_prev[:, m])
            w[m] = self.measurement_model(z, x_new, theta_sens)
            X_bar[:, m] = x_new
        X = self.resample(X_bar, w)
        self.X_prev = X

    def estimate_pose(self):
        """
        Estimates the robot's pose based on the particles and their weights
        """
        pass

    def publish_pose(self):
        """
        Publishes the estimated pose as a PoseStamped message
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose.position.x = np.mean(self.particles[0, :])
        pose_msg.pose.position.y = np.mean(self.particles[1, :])
        pose_msg.pose.orientation.z = np.mean(self.particles[2, :])
        self.pose_pub.publish(pose_msg)

    def publish_particles(self):
        """
        Publishes the particles as a MarkerArray for visualization in rviz
        """
        particle_msg = MarkerArray()
        for i in range(100):
            marker = Marker()
            marker.header.stamp = rospy.Time.now()
            marker.pose.position.x = self.particles[0, i]
            marker.pose.position.y = self.particles[1, i]
            particle_msg.markers.append(marker)
        self.particle_pub.publish(particle_msg)

    def run(self):
        rospy.spin()


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
