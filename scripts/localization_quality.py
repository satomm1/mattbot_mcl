import threading

import rospy
import rospkg
import tf
from std_msgs.msg import Bool, Int32, Float32
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import MapMetaData, Odometry

import numpy as np

LIDAR_MAX_RANGE = 16
ROBOT_MODE_IDLE = 0
ROBOT_MODE_TRACK = 4
ROBOT_MODE_PARK_HEADING = 6
ROBOT_MODE_POSE_REFINE = 13
ROBOT_MODE_RECOVERY_VALIDATE = 14


class LocalizationQuality:
    def __init__(self):
        rospy.init_node('localization_quality', anonymous=True)

        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('mattbot_mcl')
        data_path = pkg_path + '/lookup_table/current_map.npy'
        self.dist_lookup_table = np.load(data_path)

        self.lidar_measurement_skip = rospy.get_param('~lidar_measurement_skip', 2)

        map_md = rospy.wait_for_message('/map_metadata', MapMetaData)
        self.map_resolution = map_md.resolution
        self.map_width = map_md.width
        self.map_height = map_md.height

        self.localized = False
        self.localized_start_time = 0
        self.localization_warmup_sec = rospy.get_param('~localization_warmup_sec', 10.0)
        self.track_monitor_grace_sec = rospy.get_param('~track_monitor_grace_sec', 2.0)
        self.track_monitor_start_time = rospy.Time(0)

        self.max_history_length = rospy.get_param('~max_history_length', 30)
        self.max_location_history_length = rospy.get_param(
            '~recovery_history_length', 20
        )
        self.location_history = []
        # odom -> laser_frame pose for each location_history entry (None if the lookup failed);
        # carries an old map pose forward to now: anchor (+) (odom_then^-1 (+) odom_now)
        self.odom_history = []
        self.weight_history = []
        # scan_callback and robot_state_callback run on different rospy threads and share the histories
        self._lock = threading.RLock()

        self.weight_drop_factor = rospy.get_param('/weight_drop_factor', 0.5)
        self.use_absolute_score_floor = rospy.get_param(
            '~use_absolute_score_floor', False
        )
        self.mean_weight_limit = rospy.get_param('/mean_weight_limit', 2.0)
        self.recovery_cooldown_sec = rospy.get_param('~recovery_cooldown_sec', 15.0)
        self.min_improvement_ratio = rospy.get_param('~min_improvement_ratio', 1.15)
        self.max_recovery_pose_distance = rospy.get_param(
            '~max_recovery_pose_distance', 2.0
        )
        self.local_search_radius = rospy.get_param('~local_search_radius', 0.2)
        self.local_search_step = rospy.get_param('~local_search_step', 0.2)
        self.localized_laser_range_max = rospy.get_param(
            '~localized_laser_range_max', 3.0
        )
        self.pose_jump_detection_enabled = rospy.get_param(
            '~pose_jump_detection_enabled', True
        )
        self.max_pose_jump_m = rospy.get_param('~max_pose_jump_m', 1.0)
        self.max_pose_jump_theta = rospy.get_param('~max_pose_jump_theta', 0.5)
        self.post_recovery_jump_grace_sec = rospy.get_param(
            '~post_recovery_jump_grace_sec', 2.0
        )
        self.post_recovery_jump_grace_until = rospy.Time(0)

        self.recovery_anchor_radius = rospy.get_param('~recovery_anchor_radius', 5.0)
        self.recovery_orientation_delta = rospy.get_param(
            '~recovery_orientation_delta', 0.35
        )
        self.recovery_orientation_count = rospy.get_param(
            '~recovery_orientation_count', 3
        )
        self.recovery_search_verbose = rospy.get_param(
            '~recovery_search_verbose', True
        )
        self.post_backing_settle_sec = rospy.get_param('~post_backing_settle_sec', 0.3)

        self.validation_scan_count = rospy.get_param('~validation_scan_count', 3)
        self.validation_score_floor = rospy.get_param('~validation_score_floor', 1.5)
        self.validation_score_ratio = rospy.get_param('~validation_score_ratio', 0.75)
        self.max_acceptance_from_anchor = rospy.get_param(
            '~max_acceptance_from_anchor', 2.0
        )
        # Limits on AMCL's own covariance (/amcl_pose) after the post-spin stationary updates
        self.validation_max_position_std = rospy.get_param(
            '~validation_max_position_std', 0.25
        )
        self.validation_max_yaw_std = rospy.get_param('~validation_max_yaw_std', 0.25)
        # Heading vs the anchor carried forward by odometry (catches 180 deg flips); <= 0 disables
        self.validation_max_yaw_from_anchor = rospy.get_param(
            '~validation_max_yaw_from_anchor', 0.8
        )
        # Start validating at most this long after entering RECOVERY_VALIDATE, even if the
        # stationary AMCL updates have not finished (the navigator times out at 3 s)
        self.validation_settle_timeout_sec = rospy.get_param(
            '~validation_settle_timeout_sec', 1.5
        )

        # Stationary AMCL updates: AMCL only updates after update_min_d / update_min_a of motion, so
        # its pose after a spin comes from scans taken while rotating. Once /odom shows the robot at
        # rest, call request_nomotion_update a few times so AMCL refines from still scans.
        self.rest_updates_enabled = rospy.get_param('~rest_updates_enabled', True)
        self.rest_linear_threshold = rospy.get_param('~rest_linear_threshold', 0.02)
        self.rest_angular_threshold = rospy.get_param('~rest_angular_threshold', 0.05)
        self.rest_settle_sec = rospy.get_param('~rest_settle_sec', 0.25)
        self.rest_update_count = int(rospy.get_param('~rest_update_count', 3))
        self.rest_update_spacing_sec = rospy.get_param('~rest_update_spacing_sec', 0.12)
        self.nomotion_service_name = rospy.get_param(
            '~nomotion_service', '/request_nomotion_update'
        )
        # Idle / park heading correction waits this long at most for the stationary updates
        self.heading_fix_max_wait_sec = rospy.get_param('~heading_fix_max_wait_sec', 2.0)

        self.recovery_active = False
        self.recovery_anchor = None
        self.recovery_anchor_odom = None
        self.recovery_history_snapshot = []
        self.pre_loss_reference = None
        self.pose_refine_done = False
        self.pose_refine_enter_time = rospy.Time(0)
        self.validation_scores = []
        self.validation_poses = []
        self.validation_done = False
        self.validation_enter_time = rospy.Time(0)

        self.last_lost_localization_time = rospy.Time.now()

        self._load_measurement_model_params()
        self._rebuild_prob_lookup_table()

        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        self.localized_sub = rospy.Subscriber('/localized', Bool, self.localized_callback, queue_size=10)

        self.robot_state = 0
        self.made_idle_adjustment = False
        self.pending_park_heading_fix = False
        self.park_fix_ready_time = rospy.Time(0)
        self.park_settle_duration = rospy.get_param('~park_settle_duration', 0.5)
        self.heading_weight_ratio = rospy.get_param('~heading_weight_ratio', 1.15)
        self.heading_min_delta = rospy.get_param('~heading_min_delta', 0.05)
        self.robot_state_sub = rospy.Subscriber('/robot_mode', Int32, self.robot_state_callback, queue_size=10)

        self.lost_localization_pub = rospy.Publisher('/lost_localization', Bool, queue_size=10)
        self.initial_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
        self.recovery_pose_ready_pub = rospy.Publisher(
            '/recovery_pose_ready', Bool, queue_size=1, latch=True
        )
        self.recovery_validated_pub = rospy.Publisher(
            '/recovery_validated', Bool, queue_size=1, latch=True
        )
        self.scan_match_score_pub = rospy.Publisher('~scan_match_score', Float32, queue_size=1)
        self.scan_match_recent_mean_pub = rospy.Publisher(
            '~scan_match_recent_mean', Float32, queue_size=1
        )
        self.scan_match_old_mean_pub = rospy.Publisher(
            '~scan_match_old_mean', Float32, queue_size=1
        )
        self.recovery_fired_pub = rospy.Publisher(
            '~recovery_fired', Bool, queue_size=1, latch=True
        )
        self.diagnostics_log_interval = rospy.get_param('~diagnostics_log_interval', 2.0)
        self.log_scan_match_diagnostics = rospy.get_param(
            '~log_scan_match_diagnostics', False
        )
        self._last_diagnostics_log_time = rospy.Time(0)

        self.trans_listener = tf.TransformListener()

        self.idle_enter_time = rospy.Time(0)
        self.validation_start_time = None

        # Latest /amcl_pose (pose + covariance of the whole particle set) and when it arrived
        self.last_amcl_pose = None
        self.last_amcl_pose_rx = rospy.Time(0)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_pose_callback, queue_size=5)

        # Rest detector state (read and written only by the odom callback and the rest timer)
        self._odom_twist = None          # (v, w, receive time)
        self._last_motion_time = rospy.Time(0)
        self._motion_since_rest = False
        self._rest_updates_remaining = 0
        self._next_rest_update_time = rospy.Time(0)
        self._rest_batch_done_time = rospy.Time(0)
        self._nomotion_srv = rospy.ServiceProxy(self.nomotion_service_name, Empty)
        self._nomotion_warned = False
        rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=10)
        if self.rest_updates_enabled:
            self._rest_timer = rospy.Timer(rospy.Duration(0.05), self._rest_timer_callback)

        rospy.loginfo(
            "[LocalizationQuality] model z_hit=%.3f z_rand=%.3f sigma_hit=%.4f "
            "range_max=%.2f min_improvement=%.2f max_pose_dist=%.2fm "
            "pose_jump=%s max_jump=%.2fm/%.2frad",
            self.z_hit,
            self.z_random,
            self.sigma_hit,
            self.localized_laser_range_max,
            self.min_improvement_ratio,
            self.max_recovery_pose_distance,
            self.pose_jump_detection_enabled,
            self.max_pose_jump_m,
            self.max_pose_jump_theta,
        )

    # ------------------------------------------------------------------
    # Stationary AMCL updates (request_nomotion_update once the robot is at rest)
    # ------------------------------------------------------------------
    def odom_callback(self, msg):
        self._odom_twist = (
            float(msg.twist.twist.linear.x),
            float(msg.twist.twist.angular.z),
            rospy.Time.now(),
        )

    def amcl_pose_callback(self, msg):
        self.last_amcl_pose = msg
        self.last_amcl_pose_rx = rospy.Time.now()

    def _rest_timer_callback(self, _event):
        twist = self._odom_twist
        if twist is None:
            return
        v, w, rx = twist
        now = rospy.Time.now()
        if (now - rx).to_sec() > 0.5:
            return  # no fresh odometry: do not guess
        moving = abs(v) > self.rest_linear_threshold or abs(w) > self.rest_angular_threshold
        if moving:
            self._last_motion_time = now
            self._motion_since_rest = True
            self._rest_updates_remaining = 0  # moving again: drop any unfinished batch
            return
        if (
            self._motion_since_rest
            and (now - self._last_motion_time).to_sec() >= self.rest_settle_sec
        ):
            self._motion_since_rest = False
            self._rest_updates_remaining = max(0, self.rest_update_count)
            self._next_rest_update_time = now
        if self._rest_updates_remaining > 0 and now >= self._next_rest_update_time:
            self._call_nomotion_update()
            self._rest_updates_remaining -= 1
            self._next_rest_update_time = now + rospy.Duration(self.rest_update_spacing_sec)
            if self._rest_updates_remaining == 0:
                # AMCL applies the request on its next scan; allow one spacing for it to land
                self._rest_batch_done_time = self._next_rest_update_time

    def _call_nomotion_update(self):
        try:
            self._nomotion_srv()
        except (rospy.ServiceException, rospy.ROSException) as exc:
            if not self._nomotion_warned:
                rospy.logwarn(
                    '[LocalizationQuality] %s unavailable (%s); stationary AMCL updates skipped',
                    self.nomotion_service_name, exc,
                )
                self._nomotion_warned = True

    def _rest_settled(self):
        """True once the robot is at rest and the stationary AMCL updates for this stop are done."""
        if not self.rest_updates_enabled:
            return True
        twist = self._odom_twist
        if twist is None:
            return True  # no odometry: nothing to wait for
        if self._motion_since_rest or self._rest_updates_remaining > 0:
            return False
        v, w, _ = twist
        if abs(v) > self.rest_linear_threshold or abs(w) > self.rest_angular_threshold:
            return False
        return rospy.Time.now() >= self._rest_batch_done_time

    # ------------------------------------------------------------------
    # SE(2) helpers: carry an old map-frame pose forward by odometry
    # ------------------------------------------------------------------
    def _lookup_odom_pose(self):
        """odom -> laser_frame as a 3x1 array, or None."""
        try:
            trans, rot = self.trans_listener.lookupTransform('/odom', '/laser_frame', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None
        yaw = tf.transformations.euler_from_quaternion(rot)[2]
        return np.array([[trans[0]], [trans[1]], [yaw]])

    @staticmethod
    def _propagate_pose(map_pose, odom_then, odom_now):
        """map_pose (+) (odom_then^-1 (+) odom_now); returns map_pose unchanged if odom is missing."""
        if odom_then is None or odom_now is None:
            return np.array(map_pose, copy=True)
        ox, oy, oth = float(odom_then[0, 0]), float(odom_then[1, 0]), float(odom_then[2, 0])
        nx, ny, nth = float(odom_now[0, 0]), float(odom_now[1, 0]), float(odom_now[2, 0])
        c, s = np.cos(oth), np.sin(oth)
        dx_w, dy_w = nx - ox, ny - oy
        rel_x = c * dx_w + s * dy_w
        rel_y = -s * dx_w + c * dy_w
        rel_th = nth - oth
        mx, my, mth = float(map_pose[0, 0]), float(map_pose[1, 0]), float(map_pose[2, 0])
        cm, sm = np.cos(mth), np.sin(mth)
        px = mx + cm * rel_x - sm * rel_y
        py = my + sm * rel_x + cm * rel_y
        pth = np.arctan2(np.sin(mth + rel_th), np.cos(mth + rel_th))
        return np.array([[px], [py], [pth]])

    def _load_measurement_model_params(self):
        """Match AMCL / navigator laser params when available."""
        self.z_hit = float(
            rospy.get_param(
                '~z_hit',
                rospy.get_param(
                    '/amcl/laser_z_hit',
                    rospy.get_param('/z_hit', 0.75),
                ),
            )
        )
        self.z_random = float(
            rospy.get_param(
                '~z_rand',
                rospy.get_param(
                    '/amcl/laser_z_rand',
                    rospy.get_param('/z_rand', 0.25),
                ),
            )
        )
        self.sigma_hit = float(
            rospy.get_param(
                '~sigma_hit',
                rospy.get_param(
                    '/amcl/laser_sigma_hit',
                    rospy.get_param('/sigma_hit', 0.1),
                ),
            )
        )
        total = self.z_hit + self.z_random
        if total > 0:
            self.z_hit /= total
            self.z_random /= total

    def _rebuild_prob_lookup_table(self):
        unknown_indx = np.where(self.dist_lookup_table == -1)
        self.prob_lookup_table = (
            self.z_hit
            / np.sqrt(2 * np.pi * (self.sigma_hit ** 2))
            * np.exp(-0.5 * (self.dist_lookup_table) ** 2 / (self.sigma_hit ** 2))
            + self.z_random / LIDAR_MAX_RANGE
        )
        self.prob_lookup_table[unknown_indx] = 1 / LIDAR_MAX_RANGE

    def _normalize_scan_score(self, raw_score, num_valid_scans):
        if num_valid_scans > 0:
            return raw_score / num_valid_scans
        return raw_score

    def _score_pose(self, ranges, angles, x, y, theta, num_valid_scans):
        pose = np.array([[x, y, theta]]).T
        raw = self.measurement_model1(ranges, pose, angles)
        return self._normalize_scan_score(raw, num_valid_scans)

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

    def _preprocess_scan(self, msg):
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment
        range_min = msg.range_min
        range_max = self.localized_laser_range_max

        angles = np.arange(angle_min, angle_max, angle_increment)
        ranges = ranges[:: self.lidar_measurement_skip]
        angles = angles[:: self.lidar_measurement_skip]
        total_possible = len(ranges)

        valid_indx = np.where((ranges < range_max) & (ranges > range_min))
        if len(valid_indx[0]) < 200:
            valid_indx = np.where(
                (ranges < range_max + 3) & (ranges > range_min)
            )
        ranges = ranges[valid_indx]
        angles = angles[valid_indx]
        num_valid = len(ranges)
        return ranges, angles, num_valid, total_possible

    def find_best_heading(self, ranges, angles, x, y, theta_center, num_valid_scans):
        coarse_angles = np.linspace(-np.pi / 4, np.pi / 4, 100) + theta_center
        coarse_poses = np.column_stack((
            np.full(coarse_angles.shape, x),
            np.full(coarse_angles.shape, y),
            coarse_angles,
        )).T
        coarse_weights = self.measurement_model2(ranges, coarse_poses, angles)
        coarse_weights = coarse_weights / max(num_valid_scans, 1)
        coarse_best = int(np.argmax(coarse_weights))
        coarse_theta = coarse_angles[coarse_best]

        fine_angles = np.linspace(-0.1, 0.1, 50) + coarse_theta
        fine_poses = np.column_stack((
            np.full(fine_angles.shape, x),
            np.full(fine_angles.shape, y),
            fine_angles,
        )).T
        fine_weights = self.measurement_model2(ranges, fine_poses, angles)
        fine_weights = fine_weights / max(num_valid_scans, 1)
        fine_best = int(np.argmax(fine_weights))
        return fine_angles[fine_best], fine_weights[fine_best]

    def _local_search_offsets(self):
        step = self.local_search_step
        radius = self.local_search_radius
        vals = np.arange(-radius, radius + step * 0.5, step)
        offsets = []
        for dx in vals:
            for dy in vals:
                offsets.append((float(dx), float(dy)))
        return offsets

    def _orientation_candidates(self, theta_hint):
        """Small yaw set around hint for grid search (not full heading refinement)."""
        d = self.recovery_orientation_delta
        n = max(1, int(self.recovery_orientation_count))
        if n == 1:
            return [float(theta_hint)]
        out = [float(theta_hint)]
        k = 1
        while len(out) < n:
            out.append(float(theta_hint + k * d))
            if len(out) >= n:
                break
            out.append(float(theta_hint - k * d))
            k += 1
        return out[:n]

    @staticmethod
    def _xy_key(x, y, resolution):
        r = max(resolution, 1e-6)
        return (int(round(x / r)), int(round(y / r)))

    def find_best_recovery_pose_anchored(
        self,
        ranges,
        angles,
        laser_x,
        laser_y,
        laser_theta,
        anchor_pose,
        history_snapshot,
        num_valid_scans,
    ):
        """
        Post-backing recovery search: seeds near frozen anchor, expand with
        _local_search_offsets, score each grid cell at several yaws.
        """
        ax = float(anchor_pose[0, 0])
        ay = float(anchor_pose[1, 0])
        ath = float(anchor_pose[2, 0])

        # Phase 1: build seeds (anchor, history near anchor, current laser)
        seeds = [('anchor', ax, ay, ath)]
        for pose in history_snapshot:
            px = float(pose[0, 0])
            py = float(pose[1, 0])
            pth = float(pose[2, 0])
            if np.hypot(px - ax, py - ay) > self.recovery_anchor_radius:
                continue
            seeds.append(('history', px, py, pth))
        seeds.append(('current', laser_x, laser_y, laser_theta))

        current_w = self._score_pose(
            ranges, angles, laser_x, laser_y, laser_theta, num_valid_scans
        )
        best = (laser_x, laser_y, laser_theta, current_w, 'current')
        scored = []
        seen_xy = set()
        n_eval = 0

        # Phase 2–3: local grid at each seed × orientation candidates
        for prefix, sx, sy, th_hint in seeds:
            for dx, dy in self._local_search_offsets():
                cx = sx + dx
                cy = sy + dy
                key = self._xy_key(cx, cy, self.map_resolution)
                if key in seen_xy:
                    continue
                seen_xy.add(key)
                tag = prefix + '_grid' if (dx != 0.0 or dy != 0.0) else prefix
                for theta in self._orientation_candidates(th_hint):
                    w = self._score_pose(
                        ranges, angles, cx, cy, theta, num_valid_scans
                    )
                    n_eval += 1
                    scored.append((w, cx, cy, theta, tag))
                    if w > best[3]:
                        best = (cx, cy, theta, w, tag)

        self._log_recovery_search(
            ax, ay, ath, laser_x, laser_y, laser_theta,
            len(seeds), len(seen_xy), n_eval, scored, best, current_w,
        )
        return best, current_w

    def _log_recovery_search(
        self, ax, ay, ath, lx, ly, lth, n_seeds, n_cells, n_eval,
        scored, best, current_w,
    ):
        if not self.recovery_search_verbose:
            return
        required = current_w * self.min_improvement_ratio
        by_tag = {}
        for w, x, y, th, tag in scored:
            prev = by_tag.get(tag)
            if prev is None or w > prev[0]:
                by_tag[tag] = (w, x, y, th)
        top3 = sorted(scored, key=lambda t: t[0], reverse=True)[:3]
        lines = [
            '[LocalizationQuality][RecoverySearch] anchor=(%.2f,%.2f,%.2f) '
            'laser=(%.2f,%.2f,%.2f) seeds=%d cells=%d evals=%d' % (
                ax, ay, ath, lx, ly, lth, n_seeds, n_cells, n_eval,
            ),
        ]
        for tag, (w, x, y, th) in sorted(by_tag.items()):
            lines.append('  best[%s]=(%.2f,%.2f,%.2f) score=%.4f' % (tag, x, y, th, w))
        lines.append('  top3:')
        for w, x, y, th, tag in top3:
            lines.append(
                '    %.4f (%.2f,%.2f,%.2f) %s' % (w, x, y, th, tag)
            )
        bx, by, bth, bw, btag = best
        inject = bw >= required
        lines.append(
            '  decision: current=%.4f best=%.4f need=%.4f source=%s inject=%s'
            % (current_w, bw, required, btag, inject)
        )
        rospy.loginfo('\n'.join(lines))

    def _begin_recovery_session(self, anchor_pose, anchor_odom=None, pre_loss_reference=None):
        """
        anchor_pose: last trusted map-frame laser pose; anchor_odom: odom pose at that moment.
        pre_loss_reference: healthy scan-match score from before the loss (validation compares to it).
        """
        self.recovery_active = True
        self.recovery_anchor = np.array(anchor_pose, copy=True)
        self.recovery_anchor_odom = (
            None if anchor_odom is None else np.array(anchor_odom, copy=True)
        )
        self.recovery_history_snapshot = list(self.location_history)
        self.pre_loss_reference = pre_loss_reference
        self.pose_refine_done = False
        self.validation_scores = []
        self.validation_poses = []
        self.validation_done = False
        self.recovery_pose_ready_pub.publish(Bool(data=False))

    def _end_recovery_session(self):
        self.recovery_active = False
        self.recovery_anchor = None
        self.recovery_anchor_odom = None
        self.recovery_history_snapshot = []
        self.pre_loss_reference = None
        self.validation_scores = []
        self.validation_poses = []
        self.validation_done = False

    def _handle_pose_refine(
        self, ranges, angles, laser_x, laser_y, laser_theta, num_valid_scans,
    ):
        """Run one anchored search after backing; inject and signal navigator."""
        if self.pose_refine_done:
            return
        if self.pose_refine_enter_time.to_sec() == 0:
            self.pose_refine_enter_time = rospy.Time.now()
        if (
            (rospy.Time.now() - self.pose_refine_enter_time).to_sec()
            < self.post_backing_settle_sec
        ):
            return

        odom_now = self._lookup_odom_pose()
        history = []
        if self.recovery_active and self.recovery_anchor is not None:
            anchor = self.recovery_anchor
            anchor_odom = self.recovery_anchor_odom
            history = self.recovery_history_snapshot
        elif self.location_history:
            anchor = self.location_history[-1]
            anchor_odom = self.odom_history[-1]
            history = list(self.location_history)
        else:
            anchor = np.array([[laser_x], [laser_y], [laser_theta]])
            anchor_odom = odom_now

        # Keep anchor for post-spin validation (waypoint backing has no prior session).
        self.recovery_anchor = np.array(anchor, copy=True)
        self.recovery_anchor_odom = None if anchor_odom is None else np.array(anchor_odom, copy=True)

        # Search around the anchor carried forward by odometry (the robot has backed up since)
        search_anchor = self._propagate_pose(anchor, anchor_odom, odom_now)

        best, current_w = self.find_best_recovery_pose_anchored(
            ranges,
            angles,
            laser_x,
            laser_y,
            laser_theta,
            search_anchor,
            history,
            num_valid_scans,
        )
        best_x, best_y, best_th, best_w, source = best
        required = current_w * self.min_improvement_ratio

        if best_w >= required:
            rospy.loginfo(
                '[LocalizationQuality] POSE_REFINE inject from %s '
                '(%.4f -> %.4f) at (%.2f, %.2f, %.2f)',
                source, current_w, best_w, best_x, best_y, best_th,
            )
            self.initial_pose_pub.publish(
                self._make_initial_pose_msg(best_x, best_y, best_th)
            )
        else:
            rospy.logwarn(
                '[LocalizationQuality] POSE_REFINE skip inject '
                '(best=%.4f %s vs current=%.4f need=%.4f)',
                best_w, source, current_w, required,
            )

        self.pose_refine_done = True
        self.recovery_pose_ready_pub.publish(Bool(data=True))

    def _handle_recovery_validate(
        self, ranges, angles, laser_x, laser_y, laser_theta, num_valid_scans,
    ):
        """
        After the relocalize spin: wait for the stationary AMCL updates, score a few still scans,
        then publish pass/fail. Pass needs all of:
          - scan match: mean >= validation_score_floor, and >= validation_score_ratio x the healthy
            pre-loss score when known;
          - position and heading close to the anchor carried forward by odometry;
          - AMCL's own covariance (/amcl_pose after the updates) small enough.
        """
        if self.validation_done:
            return
        now = rospy.Time.now()
        if self.recovery_anchor is None:
            self.recovery_anchor = np.array(
                [[laser_x], [laser_y], [laser_theta]]
            )
            self.recovery_anchor_odom = self._lookup_odom_pose()

        if self.validation_start_time is None:
            waited = (now - self.validation_enter_time).to_sec()
            if not self._rest_settled() and waited < self.validation_settle_timeout_sec:
                return
            self.validation_start_time = now
            self.validation_scores = []
            self.validation_poses = []
            if waited >= self.validation_settle_timeout_sec and not self._rest_settled():
                rospy.logwarn(
                    '[LocalizationQuality] RECOVERY_VALIDATE: stationary AMCL updates not done '
                    'after %.1fs; validating anyway', waited,
                )

        w = self._score_pose(
            ranges, angles, laser_x, laser_y, laser_theta, num_valid_scans
        )
        self.validation_scores.append(w)
        self.validation_poses.append((laser_x, laser_y, laser_theta))

        n = self.validation_scan_count
        if len(self.validation_scores) < n:
            return

        scores = self.validation_scores[-n:]
        poses = self.validation_poses[-n:]
        mean_score = float(np.mean(scores))
        ref = self.pre_loss_reference
        score_ok = mean_score >= self.validation_score_floor
        if ref is not None:
            score_ok = score_ok and mean_score >= self.validation_score_ratio * ref

        # Where the trusted anchor should be now, by odometry
        predicted = self._propagate_pose(
            self.recovery_anchor, self.recovery_anchor_odom, self._lookup_odom_pose()
        )
        px, py, pth = float(predicted[0, 0]), float(predicted[1, 0]), float(predicted[2, 0])
        lx, ly, lth = poses[-1]
        dist = float(np.hypot(lx - px, ly - py))
        dist_ok = dist < self.max_acceptance_from_anchor
        yaw_err = float(abs(np.arctan2(np.sin(lth - pth), np.cos(lth - pth))))
        yaw_check = (
            self.validation_max_yaw_from_anchor > 0 and self.recovery_anchor_odom is not None
        )
        yaw_ok = (not yaw_check) or yaw_err < self.validation_max_yaw_from_anchor

        # AMCL covariance from an /amcl_pose published since validation began (after the updates)
        amcl = self.last_amcl_pose
        pos_std = yaw_std = float('nan')
        cov_ok = True
        cov_note = 'none'
        if amcl is not None and self.last_amcl_pose_rx >= self.validation_enter_time:
            c = np.array(amcl.pose.covariance).reshape(6, 6)
            pos_std = float(np.sqrt(max(np.linalg.eigvalsh(c[:2, :2]).max(), 0.0)))
            yaw_std = float(np.sqrt(max(c[5, 5], 0.0)))
            cov_ok = (
                pos_std < self.validation_max_position_std
                and yaw_std < self.validation_max_yaw_std
            )
            cov_note = 'fresh'
        else:
            cov_note = 'no fresh /amcl_pose, check skipped'

        passed = score_ok and dist_ok and yaw_ok and cov_ok
        self.validation_done = True
        self.recovery_validated_pub.publish(Bool(data=passed))
        details = (
            'mean_score=%.3f ref=%s score_ok=%s | dist=%.2fm dist_ok=%s | '
            'yaw_err=%.3f yaw_ok=%s%s | amcl pos_std=%.3f yaw_std=%.3f cov_ok=%s (%s)'
            % (
                mean_score, 'none' if ref is None else '%.3f' % ref, score_ok,
                dist, dist_ok, yaw_err, yaw_ok, '' if yaw_check else ' (unchecked)',
                pos_std, yaw_std, cov_ok, cov_note,
            )
        )
        if passed:
            rospy.loginfo('[LocalizationQuality] RECOVERY_VALIDATE pass (%s)', details)
            self._end_recovery_session()
        else:
            rospy.logwarn('[LocalizationQuality] RECOVERY_VALIDATE fail (%s)', details)

    def find_best_recovery_pose(
        self, ranges, angles, laser_x, laser_y, laser_theta, current_w, num_valid_scans
    ):
        """
        Search current pose, local (x,y) grid with heading refinement, and recent
        nearby history poses. Returns (x, y, theta, score, source) or None.
        """
        best = (laser_x, laser_y, laser_theta, current_w, 'current')

        for dx, dy in self._local_search_offsets():
            if dx == 0.0 and dy == 0.0:
                continue
            cx = laser_x + dx
            cy = laser_y + dy
            th, w = self.find_best_heading(
                ranges, angles, cx, cy, laser_theta, num_valid_scans
            )
            if w > best[3]:
                best = (cx, cy, th, w, 'local_search')

        for pose in self.location_history[-self.max_location_history_length :]:
            px = float(pose[0, 0])
            py = float(pose[1, 0])
            pth = float(pose[2, 0])
            if np.hypot(px - laser_x, py - laser_y) > self.max_recovery_pose_distance:
                continue
            w = self._score_pose(ranges, angles, px, py, pth, num_valid_scans)
            if w > best[3]:
                best = (px, py, pth, w, 'history')

        return best

    def _make_initial_pose_msg(self, x, y, theta):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        quat = tf.transformations.quaternion_from_euler(0.0, 0.0, theta)
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]
        covariance = np.zeros((6, 6))
        covariance[0, 0] = 0.1
        covariance[1, 1] = 0.1
        covariance[5, 5] = 3.14
        msg.pose.covariance = covariance.flatten().tolist()
        return msg

    def publish_heading_correction(self, x, y, theta):
        msg = self._make_initial_pose_msg(x, y, theta)
        covariance = np.zeros((6, 6))
        covariance[0, 0] = 0.01
        covariance[1, 1] = 0.01
        covariance[5, 5] = 0.05
        msg.pose.covariance = covariance.flatten().tolist()
        self.initial_pose_pub.publish(msg)

    def try_heading_correction(
        self, ranges, angles, laser_x, laser_y, laser_theta, current_weight, num_valid_scans, reason
    ):
        best_theta, best_weight = self.find_best_heading(
            ranges, angles, laser_x, laser_y, laser_theta, num_valid_scans
        )
        delta = abs(
            np.arctan2(
                np.sin(best_theta - laser_theta),
                np.cos(best_theta - laser_theta),
            )
        )
        if delta < self.heading_min_delta:
            return False
        if best_weight <= current_weight * self.heading_weight_ratio:
            return False

        rospy.loginfo(
            "[LocalizationQuality] %s heading correction: delta=%.3f rad, weight %.4f -> %.4f",
            reason,
            delta,
            current_weight,
            best_weight,
        )
        self.publish_heading_correction(laser_x, laser_y, best_theta)
        return True

    @staticmethod
    def _pose_delta(prev_pose, x, y, theta):
        px = float(prev_pose[0, 0])
        py = float(prev_pose[1, 0])
        pth = float(prev_pose[2, 0])
        delta_xy = float(np.hypot(x - px, y - py))
        delta_theta = abs(
            np.arctan2(
                np.sin(theta - pth),
                np.cos(theta - pth),
            )
        )
        return delta_xy, delta_theta

    def _localization_warmup_active(self):
        return (
            rospy.Time.now() - self.localized_start_time
        ).to_sec() < self.localization_warmup_sec

    def _track_monitor_grace_active(self):
        return (
            rospy.Time.now() - self.track_monitor_start_time
        ).to_sec() < self.track_monitor_grace_sec

    def _post_recovery_jump_grace_active(self):
        return rospy.Time.now() < self.post_recovery_jump_grace_until

    def _recovery_cooldown_active(self):
        return (
            rospy.Time.now() - self.last_lost_localization_time
        ).to_sec() < self.recovery_cooldown_sec

    def _mark_recovery_fired(self):
        now = rospy.Time.now()
        self.last_lost_localization_time = now
        self.post_recovery_jump_grace_until = now + rospy.Duration(
            self.post_recovery_jump_grace_sec
        )

    def _append_track_history(self, pose, weight, odom_pose):
        self.location_history.append(pose)
        self.odom_history.append(odom_pose)
        self.weight_history.append(weight)
        if len(self.weight_history) > self.max_history_length:
            self.weight_history.pop(0)
        if len(self.location_history) > self.max_location_history_length:
            self.location_history.pop(0)
            self.odom_history.pop(0)

    def _trigger_jump_recovery(
        self, last_pose, bad_x, bad_y, bad_theta, delta_xy, delta_theta, last_odom=None
    ):
        last_x = float(last_pose[0, 0])
        last_y = float(last_pose[1, 0])
        last_theta = float(last_pose[2, 0])

        self._mark_recovery_fired()
        rospy.logwarn(
            'localization_quality: pose jump detected (delta_xy=%.2fm dtheta=%.3f rad) '
            'bad=(%.2f, %.2f, %.2f) anchor=(%.2f, %.2f, %.2f); recovery after backing',
            delta_xy,
            delta_theta,
            bad_x,
            bad_y,
            bad_theta,
            last_x,
            last_y,
            last_theta,
        )
        self.recovery_fired_pub.publish(Bool(data=True))
        # Jump scans are never appended, so the latest scores are still from before the jump
        reference = (
            float(np.mean(self.weight_history[-10:]))
            if len(self.weight_history) >= 10 else None
        )
        self._begin_recovery_session(
            last_pose, anchor_odom=last_odom, pre_loss_reference=reference
        )
        self.lost_localization_pub.publish(Bool(data=True))

    def _score_window_means(self):
        """
        Compare mean of the last 10 scores vs the middle 10 of the history buffer.
        Requires a full buffer (max_history_length); shorter buffers return None.
        """
        n = len(self.weight_history)
        if n < self.max_history_length:
            return None, None
        recent_mean = float(np.mean(self.weight_history[-10:]))
        old_mean = float(np.mean(self.weight_history[10:-10]))
        return recent_mean, old_mean

    def _publish_scan_match_diagnostics(self, w_norm, recent_mean=None, old_mean=None):
        self.scan_match_score_pub.publish(Float32(data=float(w_norm)))
        if recent_mean is not None:
            self.scan_match_recent_mean_pub.publish(Float32(data=float(recent_mean)))
        if old_mean is not None:
            self.scan_match_old_mean_pub.publish(Float32(data=float(old_mean)))

        if not self.log_scan_match_diagnostics:
            return
        if self.robot_state != ROBOT_MODE_TRACK:
            return
        now = rospy.Time.now()
        if (now - self._last_diagnostics_log_time).to_sec() < self.diagnostics_log_interval:
            return
        self._last_diagnostics_log_time = now
        parts = ['scan_match_score=%.4f robot_mode=%d' % (w_norm, self.robot_state)]
        if recent_mean is not None and old_mean is not None:
            parts.append('recent_mean=%.4f old_mean=%.4f' % (recent_mean, old_mean))
        rospy.loginfo('localization_quality: %s', ' '.join(parts))

    def _trigger_recovery(
        self, ranges, angles, laser_x, laser_y, laser_theta, current_w, num_valid_scans,
        recent_mean, old_mean,
    ):
        if self._recovery_cooldown_active():
            return

        self._mark_recovery_fired()
        rospy.logwarn(
            'localization_quality: lost localization (recent_mean=%.4f old_mean=%.4f '
            'drop_factor=%.2f)',
            recent_mean,
            old_mean,
            self.weight_drop_factor,
        )
        self.recovery_fired_pub.publish(Bool(data=True))

        # Anchor on the last pose before the low-score window (the newest 10 scans scored badly);
        # location_history and weight_history are aligned at their ends.
        if len(self.location_history) >= 11:
            anchor = self.location_history[-11]
            anchor_odom = self.odom_history[-11]
        elif self.location_history:
            anchor = self.location_history[0]
            anchor_odom = self.odom_history[0]
        else:
            anchor = np.array([[laser_x], [laser_y], [laser_theta]])
            anchor_odom = self._lookup_odom_pose()
        # Validate against the healthy score from before the drop, not the degraded recent mean
        self._begin_recovery_session(
            anchor, anchor_odom=anchor_odom, pre_loss_reference=old_mean
        )
        self.lost_localization_pub.publish(Bool(data=True))

    def scan_callback(self, msg):
        """
        The callback for the laser scan subscriber. This function is called whenever a new laser scan message is
        received. The function updates the particles based on the laser scan data using the measurement model. The
        particles are then resampled based on the weights calculated from the measurement model.

        Args:
            msg: LaserScan message
        """
        with self._lock:
            self._scan_callback_locked(msg)

    def _idle_heading_fix_pending(self):
        return self.robot_state == ROBOT_MODE_IDLE and (
            self.pending_park_heading_fix or not self.made_idle_adjustment
        )

    def _scan_callback_locked(self, msg):
        if not self.localized:
            return

        # Idle / park heading correction is a fallback: let AMCL refine from still scans first
        # (stationary updates), then check whether a heading correction is still worth it.
        if self._idle_heading_fix_pending():
            now = rospy.Time.now()
            if self.pending_park_heading_fix and now < self.park_fix_ready_time:
                return
            waited = (now - self.idle_enter_time).to_sec()
            if not self._rest_settled() and waited < self.heading_fix_max_wait_sec:
                return

        ranges, angles, num_valid_scans, total_possible = self._preprocess_scan(msg)
        if num_valid_scans == 0:
            return

        try:
            translation, rotation = self.trans_listener.lookupTransform(
                '/map', '/laser_frame', rospy.Time(0)
            )
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        laser_x = translation[0]
        laser_y = translation[1]
        laser_theta = tf.transformations.euler_from_quaternion(rotation)[2]
        current_pose = np.array([[laser_x, laser_y, laser_theta]]).T
        current_odom = self._lookup_odom_pose()

        w = self._score_pose(
            ranges, angles, laser_x, laser_y, laser_theta, num_valid_scans
        )

        if self._idle_heading_fix_pending():
            reason = 'park' if self.pending_park_heading_fix else 'idle'
            self.try_heading_correction(
                ranges, angles, laser_x, laser_y, laser_theta, w, num_valid_scans, reason
            )
            self.made_idle_adjustment = True
            self.pending_park_heading_fix = False
            return

        # Post-backing pose search (navigator mode POSE_REFINE)
        if self.robot_state == ROBOT_MODE_POSE_REFINE:
            self._handle_pose_refine(
                ranges, angles, laser_x, laser_y, laser_theta, num_valid_scans
            )
            return

        # Post-spin validation (navigator mode RECOVERY_VALIDATE)
        if self.robot_state == ROBOT_MODE_RECOVERY_VALIDATE:
            self._handle_recovery_validate(
                ranges, angles, laser_x, laser_y, laser_theta, num_valid_scans
            )
            return

        if self.robot_state != ROBOT_MODE_TRACK:
            self._publish_scan_match_diagnostics(w, None, None)
            return

        if self.pose_jump_detection_enabled and len(self.location_history) > 0:
            delta_xy, delta_theta = self._pose_delta(
                self.location_history[-1], laser_x, laser_y, laser_theta
            )
            pose_jump = (
                delta_xy > self.max_pose_jump_m
                or delta_theta > self.max_pose_jump_theta
            )
            if pose_jump:
                can_recover = (
                    not self._localization_warmup_active()
                    and not self._track_monitor_grace_active()
                    and not self._post_recovery_jump_grace_active()
                    and not self._recovery_cooldown_active()
                )
                if can_recover:
                    self._trigger_jump_recovery(
                        self.location_history[-1],
                        laser_x,
                        laser_y,
                        laser_theta,
                        delta_xy,
                        delta_theta,
                        last_odom=self.odom_history[-1],
                    )
                else:
                    rospy.logwarn(
                        'localization_quality: pose jump detected '
                        '(delta_xy=%.2fm dtheta=%.3f rad) but recovery suppressed '
                        '(warmup/grace/cooldown)',
                        delta_xy,
                        delta_theta,
                    )
                return

        self._append_track_history(current_pose, w, current_odom)
        recent_mean, old_mean = self._score_window_means()
        self._publish_scan_match_diagnostics(w, recent_mean, old_mean)

        if self._localization_warmup_active():
            return
        if len(self.weight_history) < self.max_history_length:
            return
        if self._track_monitor_grace_active():
            return

        if recent_mean is None:
            return

        score_drop = recent_mean < self.weight_drop_factor * old_mean
        score_floor = (
            self.use_absolute_score_floor
            and recent_mean < self.mean_weight_limit
        )
        if not (score_drop or score_floor):
            return

        self._trigger_recovery(
            ranges,
            angles,
            laser_x,
            laser_y,
            laser_theta,
            w,
            num_valid_scans,
            recent_mean,
            old_mean,
        )

    def robot_state_callback(self, msg):
        with self._lock:
            self._robot_state_callback_locked(msg)

    def _robot_state_callback_locked(self, msg):
        prev_state = self.robot_state
        if msg.data == ROBOT_MODE_IDLE and prev_state != ROBOT_MODE_IDLE:
            self.idle_enter_time = rospy.Time.now()
        if prev_state == ROBOT_MODE_PARK_HEADING and msg.data == ROBOT_MODE_IDLE:
            self.pending_park_heading_fix = True
            self.park_fix_ready_time = rospy.Time.now() + rospy.Duration(
                self.park_settle_duration
            )
            self.made_idle_adjustment = False
            rospy.loginfo(
                '[LocalizationQuality] Park complete; heading correction scheduled in %.1fs',
                self.park_settle_duration,
            )
        elif msg.data == 0 and prev_state != 0:
            self.made_idle_adjustment = False
        if msg.data == ROBOT_MODE_POSE_REFINE and prev_state != ROBOT_MODE_POSE_REFINE:
            self.pose_refine_enter_time = rospy.Time.now()
            self.pose_refine_done = False
            rospy.loginfo(
                '[LocalizationQuality] POSE_REFINE entered; search after %.1fs settle',
                self.post_backing_settle_sec,
            )
        if (
            msg.data == ROBOT_MODE_RECOVERY_VALIDATE
            and prev_state != ROBOT_MODE_RECOVERY_VALIDATE
        ):
            self.validation_enter_time = rospy.Time.now()
            self.validation_start_time = None
            self.validation_scores = []
            self.validation_poses = []
            self.validation_done = False
            rospy.loginfo(
                '[LocalizationQuality] RECOVERY_VALIDATE entered; waiting for stationary AMCL '
                'updates (max %.1fs), then collecting %d scans',
                self.validation_settle_timeout_sec,
                self.validation_scan_count,
            )
        if prev_state != ROBOT_MODE_TRACK and msg.data == ROBOT_MODE_TRACK:
            self.track_monitor_start_time = rospy.Time.now()
            self.weight_history = []
            self.location_history = []
            self.odom_history = []
            if self.log_scan_match_diagnostics:
                rospy.loginfo(
                    '[LocalizationQuality] TRACK started; score history cleared, '
                    '%.1fs monitor grace',
                    self.track_monitor_grace_sec,
                )
        self.robot_state = msg.data

    def localized_callback(self, msg):
        if not self.localized and msg.data:
            rospy.loginfo('[LocalizationQuality] Robot is localized')
            self.localized_start_time = rospy.Time.now()
            self.localized = True
            self.made_idle_adjustment = False

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        LocalizationQuality().run()
    except rospy.ROSInterruptException:
        pass
