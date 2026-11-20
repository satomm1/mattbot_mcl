"""
localization_quality: watches AMCL during navigation and drives localization recovery.

- Scores each lidar scan against the map at AMCL's pose (likelihood field, sum of p^3 per beam;
  a perfect beam scores ~16).
- TRACK: detects AMCL jumps (sudden map->odom corrections) and score drops, then publishes
  /lost_localization.
- Recovery (navigator modes): POSE_REFINE searches around the last trusted pose carried forward by
  odometry and injects it into AMCL; RECOVERY_VALIDATE checks the result while stationary;
  RELOCALIZING (the spin, only if validation failed) validates continuously and ends the spin early.
- Stationary AMCL updates (request_nomotion_update) whenever the robot comes to rest.
- IDLE / park heading correction as a fallback after those updates.
"""
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
ROBOT_MODE_RELOCALIZING = 9
ROBOT_MODE_POSE_REFINE = 13
ROBOT_MODE_RECOVERY_VALIDATE = 14
RECOVERY_MODES = (ROBOT_MODE_RELOCALIZING, ROBOT_MODE_POSE_REFINE, ROBOT_MODE_RECOVERY_VALIDATE)

# Recovery session kinds
SESSION_JUMP = 'jump'      # AMCL jumped: the pose it jumped to is not trusted at all
SESSION_DROP = 'drop'      # scan-match score dropped
SESSION_MOTION = 'motion'  # navigator backed up (stall, missed waypoint, ...): localization probably fine

TF_WAIT_SEC = 0.06


def _wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


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
        if self.dist_lookup_table.shape != (self.map_height, self.map_width):
            msg = (
                '[LocalizationQuality] lookup table %s has shape %s but /map_metadata is %dx%d '
                '(height x width): it belongs to another map. Re-run finalize_map.py / '
                'generate_dds_map.py for the current map.'
                % (data_path, self.dist_lookup_table.shape, self.map_height, self.map_width)
            )
            rospy.logfatal(msg)
            raise SystemExit(msg)

        self.localized = False
        self.localized_start_time = rospy.Time(0)
        self.localization_warmup_sec = rospy.get_param('~localization_warmup_sec', 10.0)
        self.track_monitor_grace_sec = rospy.get_param('~track_monitor_grace_sec', 2.0)
        self.track_monitor_start_time = rospy.Time(0)

        self.max_history_length = rospy.get_param('~max_history_length', 30)
        self.max_location_history_length = rospy.get_param('~recovery_history_length', 20)
        self.location_history = []
        # odom -> laser_frame pose for each location_history entry (None if the lookup failed);
        # carries an old map pose forward to now: anchor (+) (odom_then^-1 (+) odom_now)
        self.odom_history = []
        # Scores aligned with the end of location_history; NaN for scans scored with the range
        # fallback (their scale differs, so the drop detector skips them)
        self.weight_history = []
        self._last_map_odom = None  # map -> odom at the previous TRACK scan (jump detection)
        # scan_callback and robot_state_callback run on different rospy threads and share state
        self._lock = threading.RLock()

        # Private params; the global ones are the old names (kept as fallback)
        self.weight_drop_factor = rospy.get_param(
            '~weight_drop_factor', rospy.get_param('/weight_drop_factor', 0.6)
        )
        self.use_absolute_score_floor = rospy.get_param('~use_absolute_score_floor', False)
        self.mean_weight_limit = rospy.get_param(
            '~mean_weight_limit', rospy.get_param('/mean_weight_limit', 2.0)
        )
        self.recovery_cooldown_sec = rospy.get_param('~recovery_cooldown_sec', 15.0)
        self.min_improvement_ratio = rospy.get_param('~min_improvement_ratio', 1.15)
        self.local_search_radius = rospy.get_param('~local_search_radius', 0.2)
        self.local_search_step = rospy.get_param('~local_search_step', 0.2)
        self.localized_laser_range_max = rospy.get_param('~localized_laser_range_max', 3.0)
        self.min_valid_beams = int(rospy.get_param('~min_valid_beams', 200))
        self.pose_jump_detection_enabled = rospy.get_param('~pose_jump_detection_enabled', True)
        self.max_pose_jump_m = rospy.get_param('~max_pose_jump_m', 1.0)
        self.max_pose_jump_theta = rospy.get_param('~max_pose_jump_theta', 0.5)
        self.post_recovery_jump_grace_sec = rospy.get_param('~post_recovery_jump_grace_sec', 2.0)
        self.post_recovery_jump_grace_until = rospy.Time(0)

        # Coarse anchored search, then a fine local refinement of the best candidate
        self.recovery_anchor_radius = rospy.get_param('~recovery_anchor_radius', 5.0)
        self.recovery_orientation_delta = rospy.get_param('~recovery_orientation_delta', 0.35)
        self.recovery_orientation_count = rospy.get_param('~recovery_orientation_count', 3)
        self.refine_xy_radius = rospy.get_param('~refine_xy_radius', 0.1)
        self.refine_xy_step = rospy.get_param('~refine_xy_step', 0.05)
        self.refine_yaw_radius = rospy.get_param('~refine_yaw_radius', 0.2)
        self.refine_yaw_step = rospy.get_param('~refine_yaw_step', 0.025)
        self.recovery_search_verbose = rospy.get_param('~recovery_search_verbose', True)
        self.post_backing_settle_sec = rospy.get_param('~post_backing_settle_sec', 0.3)
        # POSE_REFINE waits (up to this long) for the robot to be still before searching
        self.refine_max_wait_sec = rospy.get_param('~refine_max_wait_sec', 1.0)
        # Covariance of the pose injected into AMCL after a recovery search. Matches the search's
        # accuracy, so AMCL keeps the heading and does not need a spin to find it again.
        self.recovery_inject_xy_std = rospy.get_param('~recovery_inject_xy_std', 0.15)
        self.recovery_inject_yaw_std = rospy.get_param('~recovery_inject_yaw_std', 0.2)

        self.validation_scan_count = rospy.get_param('~validation_scan_count', 3)
        self.validation_score_floor = rospy.get_param('~validation_score_floor', 1.5)
        self.validation_score_ratio = rospy.get_param('~validation_score_ratio', 0.7)
        self.max_acceptance_from_anchor = rospy.get_param('~max_acceptance_from_anchor', 2.0)
        # Limits on AMCL's own covariance (/amcl_pose)
        self.validation_max_position_std = rospy.get_param('~validation_max_position_std', 0.25)
        self.validation_max_yaw_std = rospy.get_param('~validation_max_yaw_std', 0.25)
        # Heading vs the anchor carried forward by odometry (catches 180 deg flips); <= 0 disables
        self.validation_max_yaw_from_anchor = rospy.get_param('~validation_max_yaw_from_anchor', 0.8)
        # Start validating at most this long after entering RECOVERY_VALIDATE, even if the
        # stationary AMCL updates have not finished (the navigator times out at 3 s)
        self.validation_settle_timeout_sec = rospy.get_param('~validation_settle_timeout_sec', 1.5)
        # RELOCALIZING (spin): consecutive passing scans that end the spin early
        self.spin_validation_scans = int(rospy.get_param('~spin_validation_scans', 5))

        # Stationary AMCL updates: AMCL only updates after update_min_d / update_min_a of motion, so
        # after a stop its pose comes from the last moving scans. Once /odom shows the robot at
        # rest, call request_nomotion_update a few times so AMCL refines from still scans.
        self.rest_updates_enabled = rospy.get_param('~rest_updates_enabled', True)
        self.rest_linear_threshold = rospy.get_param('~rest_linear_threshold', 0.02)
        self.rest_angular_threshold = rospy.get_param('~rest_angular_threshold', 0.05)
        self.rest_settle_sec = rospy.get_param('~rest_settle_sec', 0.25)
        self.rest_update_count = int(rospy.get_param('~rest_update_count', 3))
        self.rest_update_spacing_sec = rospy.get_param('~rest_update_spacing_sec', 0.12)
        self.nomotion_service_name = rospy.get_param('~nomotion_service', '/request_nomotion_update')
        # Idle / park heading correction waits this long at most for the stationary updates
        self.heading_fix_max_wait_sec = rospy.get_param('~heading_fix_max_wait_sec', 2.0)

        # Recovery session
        self.recovery_active = False
        self.recovery_kind = None
        self.recovery_anchor = None
        self.recovery_anchor_odom = None
        self.recovery_history_snapshot = []
        self.pre_loss_reference = None
        self.pose_refine_done = False
        self.pose_refine_enter_time = rospy.Time(0)
        self.validation_scores = []
        self.validation_poses = []
        self.validation_fallback = False
        self.validation_done = False
        self.validation_enter_time = rospy.Time(0)
        self.validation_start_time = None
        self.spin_enter_time = rospy.Time(0)
        self.spin_good_scans = 0
        self.spin_validated = False

        self.last_lost_localization_time = rospy.Time.now()

        self._load_measurement_model_params()
        self._rebuild_prob_lookup_table()

        self.robot_state = 0
        self.made_idle_adjustment = False
        self.pending_park_heading_fix = False
        self.park_fix_ready_time = rospy.Time(0)
        self.park_settle_duration = rospy.get_param('~park_settle_duration', 0.5)
        self.heading_weight_ratio = rospy.get_param('~heading_weight_ratio', 1.15)
        self.heading_min_delta = rospy.get_param('~heading_min_delta', 0.05)
        self.idle_enter_time = rospy.Time(0)

        self.lost_localization_pub = rospy.Publisher('/lost_localization', Bool, queue_size=10)
        self.initial_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
        self.recovery_pose_ready_pub = rospy.Publisher('/recovery_pose_ready', Bool, queue_size=1, latch=True)
        self.recovery_validated_pub = rospy.Publisher('/recovery_validated', Bool, queue_size=1, latch=True)
        self.scan_match_score_pub = rospy.Publisher('~scan_match_score', Float32, queue_size=1)
        self.scan_match_recent_mean_pub = rospy.Publisher('~scan_match_recent_mean', Float32, queue_size=1)
        self.scan_match_old_mean_pub = rospy.Publisher('~scan_match_old_mean', Float32, queue_size=1)
        self.recovery_fired_pub = rospy.Publisher('~recovery_fired', Bool, queue_size=1, latch=True)
        self.diagnostics_log_interval = rospy.get_param('~diagnostics_log_interval', 2.0)
        self.log_scan_match_diagnostics = rospy.get_param('~log_scan_match_diagnostics', False)
        self._last_diagnostics_log_time = rospy.Time(0)

        self.trans_listener = tf.TransformListener()

        # Latest /amcl_pose (pose + covariance of the whole particle set) and when it arrived
        self.last_amcl_pose = None
        self.last_amcl_pose_rx = rospy.Time(0)

        # Rest detector state (odom callback + rest timer; RECOVERY_VALIDATE may request a batch)
        self._odom_twist = None          # (v, w, receive time)
        self._last_motion_time = rospy.Time(0)
        self._motion_since_rest = False
        self._rest_updates_remaining = 0
        self._next_rest_update_time = rospy.Time(0)
        self._rest_batch_done_time = rospy.Time(0)
        self._nomotion_srv = rospy.ServiceProxy(self.nomotion_service_name, Empty)
        self._nomotion_warned = False

        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_pose_callback, queue_size=5)
        rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=10)
        self.robot_state_sub = rospy.Subscriber('/robot_mode', Int32, self.robot_state_callback, queue_size=10)
        self.localized_sub = rospy.Subscriber('/localized', Bool, self.localized_callback, queue_size=10)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        if self.rest_updates_enabled:
            self._rest_timer = rospy.Timer(rospy.Duration(0.05), self._rest_timer_callback)

        rospy.loginfo(
            '[LocalizationQuality] model z_hit=%.3f z_rand=%.3f sigma_hit=%.4f range_max=%.2f '
            'drop_factor=%.2f pose_jump=%s max_jump=%.2fm/%.2frad inject_std=%.2fm/%.2frad '
            'rest_updates=%s',
            self.z_hit, self.z_random, self.sigma_hit, self.localized_laser_range_max,
            self.weight_drop_factor, self.pose_jump_detection_enabled, self.max_pose_jump_m,
            self.max_pose_jump_theta, self.recovery_inject_xy_std, self.recovery_inject_yaw_std,
            self.rest_updates_enabled,
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

    def _robot_moving(self):
        twist = self._odom_twist
        if twist is None:
            return False
        v, w, _ = twist
        return abs(v) > self.rest_linear_threshold or abs(w) > self.rest_angular_threshold

    def _rest_timer_callback(self, _event):
        twist = self._odom_twist
        if twist is None:
            return
        now = rospy.Time.now()
        if (now - twist[2]).to_sec() > 0.5:
            return  # no fresh odometry: do not guess
        if self._robot_moving():
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

    def _request_rest_batch(self):
        """Ask for a fresh batch of stationary updates (e.g. after a pose was injected)."""
        if not self.rest_updates_enabled:
            return
        self._rest_updates_remaining = max(0, self.rest_update_count)
        self._next_rest_update_time = rospy.Time.now()

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
        if self._odom_twist is None:
            return True  # no odometry: nothing to wait for
        if self._motion_since_rest or self._rest_updates_remaining > 0 or self._robot_moving():
            return False
        return rospy.Time.now() >= self._rest_batch_done_time

    # ------------------------------------------------------------------
    # TF and SE(2) helpers
    # ------------------------------------------------------------------
    def _lookup_pose(self, target, source, stamp):
        """target -> source as a 3x1 array at stamp (latest if unavailable there), or None."""
        trans = rot = None
        if stamp is not None and stamp.to_sec() > 0:
            try:
                self.trans_listener.waitForTransform(target, source, stamp, rospy.Duration(TF_WAIT_SEC))
                trans, rot = self.trans_listener.lookupTransform(target, source, stamp)
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                trans = None
        if trans is None:
            try:
                trans, rot = self.trans_listener.lookupTransform(target, source, rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                return None
        yaw = tf.transformations.euler_from_quaternion(rot)[2]
        return np.array([[trans[0]], [trans[1]], [yaw]])

    def _lookup_odom_pose(self, stamp=None):
        """odom -> laser_frame as a 3x1 array, or None."""
        return self._lookup_pose('/odom', '/laser_frame', stamp)

    @staticmethod
    def _compose(a, b):
        """a (+) b for 3x1 SE(2) poses."""
        ax, ay, ath = float(a[0, 0]), float(a[1, 0]), float(a[2, 0])
        bx, by, bth = float(b[0, 0]), float(b[1, 0]), float(b[2, 0])
        c, s = np.cos(ath), np.sin(ath)
        return np.array([[ax + c * bx - s * by], [ay + s * bx + c * by], [_wrap(ath + bth)]])

    @staticmethod
    def _propagate_pose(map_pose, odom_then, odom_now):
        """map_pose (+) (odom_then^-1 (+) odom_now); returns map_pose unchanged if odom is missing."""
        if odom_then is None or odom_now is None:
            return np.array(map_pose, copy=True)
        ox, oy, oth = float(odom_then[0, 0]), float(odom_then[1, 0]), float(odom_then[2, 0])
        nx, ny, nth = float(odom_now[0, 0]), float(odom_now[1, 0]), float(odom_now[2, 0])
        c, s = np.cos(oth), np.sin(oth)
        dx_w, dy_w = nx - ox, ny - oy
        rel = np.array([[c * dx_w + s * dy_w], [-s * dx_w + c * dy_w], [nth - oth]])
        return LocalizationQuality._compose(map_pose, rel)

    @staticmethod
    def _pose_delta(prev_pose, x, y, theta):
        px, py, pth = float(prev_pose[0, 0]), float(prev_pose[1, 0]), float(prev_pose[2, 0])
        return float(np.hypot(x - px, y - py)), abs(_wrap(theta - pth))

    # ------------------------------------------------------------------
    # Measurement model
    # ------------------------------------------------------------------
    def _load_measurement_model_params(self):
        """Match AMCL's laser params when available."""
        self.z_hit = float(rospy.get_param('~z_hit', rospy.get_param('/amcl/laser_z_hit', 0.95)))
        self.z_random = float(rospy.get_param('~z_rand', rospy.get_param('/amcl/laser_z_rand', 0.05)))
        self.sigma_hit = float(rospy.get_param('~sigma_hit', rospy.get_param('/amcl/laser_sigma_hit', 0.15)))
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

    def _score_pose(self, ranges, angles, x, y, theta, num_valid_scans):
        """Mean p^3 per beam at one pose."""
        pose = np.array([[x], [y], [theta]])
        return float(self.measurement_model(ranges, pose, angles)[0]) / max(num_valid_scans, 1)

    def _score_poses(self, ranges, angles, poses, num_valid_scans):
        """Mean p^3 per beam for a 3xM array of poses."""
        return self.measurement_model(ranges, poses, angles) / max(num_valid_scans, 1)

    def measurement_model(self, z, x, theta_sens):
        """
        Likelihood-field model (Probabilistic Robotics, Table 6.3) for M poses at once.

        Args:
            z: ranges (N,), already limited to the valid range
            x: poses, 3xM array (x, y, theta) of the laser in the map frame
            theta_sens: beam angles in the laser frame (N,)

        Returns:
            (M,) sum over beams of p^3 (AMCL's heuristic instead of a product)
        """
        x_meas = x[0][:, None] + z[None, :] * np.cos(x[2][:, None] + theta_sens[None, :])
        y_meas = x[1][:, None] + z[None, :] * np.sin(x[2][:, None] + theta_sens[None, :])
        x_grid = np.round(x_meas / self.map_resolution).astype(int)
        y_grid = np.round(y_meas / self.map_resolution).astype(int)
        outside = (x_grid < 0) | (x_grid >= self.map_width) | (y_grid < 0) | (y_grid >= self.map_height)
        p = self.prob_lookup_table[
            np.clip(y_grid, 0, self.map_height - 1), np.clip(x_grid, 0, self.map_width - 1)
        ]
        p[outside] = 1 / LIDAR_MAX_RANGE  # off the map: unknown
        return np.sum(p ** 3, axis=1)

    def _preprocess_scan(self, msg):
        """
        Returns (ranges, angles, num_valid, used_fallback). Beams up to localized_laser_range_max;
        if fewer than min_valid_beams, up to +3 m (used_fallback: scores then have another scale).
        """
        ranges_all = np.asarray(msg.ranges, dtype=float)
        angles_all = msg.angle_min + np.arange(len(ranges_all)) * msg.angle_increment
        ranges = ranges_all[:: self.lidar_measurement_skip]
        angles = angles_all[:: self.lidar_measurement_skip]
        range_min = msg.range_min
        range_max = self.localized_laser_range_max

        valid = (ranges < range_max) & (ranges > range_min)
        used_fallback = False
        if np.count_nonzero(valid) < self.min_valid_beams:
            valid = (ranges < range_max + 3) & (ranges > range_min)
            used_fallback = True
        return ranges[valid], angles[valid], int(np.count_nonzero(valid)), used_fallback

    # ------------------------------------------------------------------
    # Pose search
    # ------------------------------------------------------------------
    def find_best_heading(self, ranges, angles, x, y, theta_center, num_valid_scans):
        coarse = np.linspace(-np.pi / 4, np.pi / 4, 100) + theta_center
        w = self._score_poses(ranges, angles, np.vstack([np.full_like(coarse, x), np.full_like(coarse, y), coarse]), num_valid_scans)
        fine = np.linspace(-0.1, 0.1, 50) + coarse[int(np.argmax(w))]
        w = self._score_poses(ranges, angles, np.vstack([np.full_like(fine, x), np.full_like(fine, y), fine]), num_valid_scans)
        i = int(np.argmax(w))
        return float(fine[i]), float(w[i])

    def _local_search_offsets(self):
        step = self.local_search_step
        radius = self.local_search_radius
        vals = np.arange(-radius, radius + step * 0.5, step)
        return [(float(dx), float(dy)) for dx in vals for dy in vals]

    def _orientation_candidates(self, theta_hint):
        """Small yaw set around hint for the coarse grid search."""
        d = self.recovery_orientation_delta
        n = max(1, int(self.recovery_orientation_count))
        out = [float(theta_hint)]
        k = 1
        while len(out) < n:
            out.append(float(theta_hint + k * d))
            if len(out) < n:
                out.append(float(theta_hint - k * d))
            k += 1
        return out[:n]

    def find_best_recovery_pose_anchored(
        self, ranges, angles, laser_x, laser_y, laser_theta, anchor_pose, history_snapshot,
        num_valid_scans, include_current=True,
    ):
        """
        Coarse search: a local grid at several yaws around each seed (the anchor, history poses near
        it and, unless include_current is False, AMCL's current pose), then a fine xy x yaw
        refinement around the best candidate. Returns ((x, y, theta, score, source), current_score).
        """
        ax, ay, ath = float(anchor_pose[0, 0]), float(anchor_pose[1, 0]), float(anchor_pose[2, 0])
        seeds = [('anchor', ax, ay, ath)]
        for pose in history_snapshot:
            px, py, pth = float(pose[0, 0]), float(pose[1, 0]), float(pose[2, 0])
            if np.hypot(px - ax, py - ay) <= self.recovery_anchor_radius:
                seeds.append(('history', px, py, pth))
        if include_current:
            seeds.append(('current', laser_x, laser_y, laser_theta))

        current_w = self._score_pose(ranges, angles, laser_x, laser_y, laser_theta, num_valid_scans)

        cand, tags, seen = [], [], set()
        r = max(self.map_resolution, 1e-6)
        for prefix, sx, sy, th_hint in seeds:
            for dx, dy in self._local_search_offsets():
                cx, cy = sx + dx, sy + dy
                key = (int(round(cx / r)), int(round(cy / r)))
                if key in seen:
                    continue
                seen.add(key)
                tag = prefix + '_grid' if (dx != 0.0 or dy != 0.0) else prefix
                for theta in self._orientation_candidates(th_hint):
                    cand.append((cx, cy, theta))
                    tags.append(tag)
        poses = np.array(cand).T
        w = self._score_poses(ranges, angles, poses, num_valid_scans)
        i = int(np.argmax(w))
        coarse = (float(poses[0, i]), float(poses[1, i]), float(poses[2, i]), float(w[i]), tags[i])

        best = self._refine_pose(ranges, angles, coarse, num_valid_scans)
        if include_current and current_w > best[3]:
            best = (laser_x, laser_y, laser_theta, current_w, 'current')

        if self.recovery_search_verbose:
            order = np.argsort(-w)[:3]
            rospy.loginfo(
                '[LocalizationQuality][RecoverySearch] anchor=(%.2f,%.2f,%.2f) laser=(%.2f,%.2f,%.2f) '
                'seeds=%d evals=%d current=%.3f coarse=%.3f %s (%.2f,%.2f,%.2f) -> refined=%.3f '
                '(%.2f,%.2f,%.2f) | top3 coarse: %s',
                ax, ay, ath, laser_x, laser_y, laser_theta, len(seeds), len(cand), current_w,
                coarse[3], coarse[4], coarse[0], coarse[1], coarse[2], best[3], best[0], best[1], best[2],
                ', '.join('%.3f %s' % (w[k], tags[k]) for k in order),
            )
        return best, current_w

    def _refine_pose(self, ranges, angles, seed, num_valid_scans):
        """Fine xy x yaw grid around seed (x, y, th, w, tag); returns the best in the same form."""
        sx, sy, sth, sw, tag = seed
        xy = np.arange(-self.refine_xy_radius, self.refine_xy_radius + 1e-9, self.refine_xy_step)
        yaw = np.arange(-self.refine_yaw_radius, self.refine_yaw_radius + 1e-9, self.refine_yaw_step)
        gx, gy, gt = np.meshgrid(sx + xy, sy + xy, sth + yaw, indexing='ij')
        poses = np.vstack([gx.ravel(), gy.ravel(), gt.ravel()])
        w = self._score_poses(ranges, angles, poses, num_valid_scans)
        i = int(np.argmax(w))
        if w[i] <= sw:
            return seed
        return (float(poses[0, i]), float(poses[1, i]), _wrap(poses[2, i]), float(w[i]), tag + '+refined')

    def _make_initial_pose_msg(self, x, y, theta, xy_std, yaw_std, stamp=None):
        msg = PoseWithCovarianceStamped()
        # Stamp with the scan time the pose belongs to: AMCL carries it forward by odometry to now
        msg.header.stamp = stamp if stamp is not None and stamp.to_sec() > 0 else rospy.Time.now()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        quat = tf.transformations.quaternion_from_euler(0.0, 0.0, theta)
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]
        covariance = np.zeros((6, 6))
        covariance[0, 0] = covariance[1, 1] = xy_std ** 2
        covariance[5, 5] = yaw_std ** 2
        msg.pose.covariance = covariance.flatten().tolist()
        return msg

    # ------------------------------------------------------------------
    # Recovery session
    # ------------------------------------------------------------------
    def _recent_reference(self):
        """Mean of the latest 10 (finite) TRACK scores, or None."""
        recent = [w for w in self.weight_history[-10:] if np.isfinite(w)]
        return float(np.mean(recent)) if len(recent) >= 5 else None

    def _begin_recovery_session(self, kind, anchor_pose, anchor_odom=None, pre_loss_reference=None):
        """
        kind: SESSION_JUMP / SESSION_DROP / SESSION_MOTION.
        anchor_pose: last trusted map-frame laser pose; anchor_odom: odom pose at that moment.
        pre_loss_reference: healthy scan-match score from before the loss (validation compares to it).
        """
        self.recovery_active = True
        self.recovery_kind = kind
        self.recovery_anchor = np.array(anchor_pose, copy=True)
        self.recovery_anchor_odom = None if anchor_odom is None else np.array(anchor_odom, copy=True)
        self.recovery_history_snapshot = list(self.location_history)
        self.pre_loss_reference = pre_loss_reference
        self.pose_refine_done = False
        self.validation_scores = []
        self.validation_poses = []
        self.validation_done = False
        self.spin_validated = False
        self.recovery_pose_ready_pub.publish(Bool(data=False))

    def _end_recovery_session(self):
        self.recovery_active = False
        self.recovery_kind = None
        self.recovery_anchor = None
        self.recovery_anchor_odom = None
        self.recovery_history_snapshot = []
        self.pre_loss_reference = None
        self.validation_scores = []
        self.validation_poses = []
        self.validation_done = False

    def _handle_pose_refine(self, ranges, angles, laser_x, laser_y, laser_theta, num_valid_scans, stamp):
        """One search once the robot is still; inject the result and signal the navigator."""
        if self.pose_refine_done:
            return
        elapsed = (rospy.Time.now() - self.pose_refine_enter_time).to_sec()
        if elapsed < self.post_backing_settle_sec:
            return
        if self._robot_moving() and elapsed < self.refine_max_wait_sec:
            return

        odom_now = self._lookup_odom_pose(stamp)
        if not self.recovery_active:
            # Navigator-initiated (backed up after a stall / missed waypoint): anchor on the last
            # TRACK pose; localization is probably fine, so only inject a clearly better pose
            if self.location_history:
                anchor, anchor_odom = self.location_history[-1], self.odom_history[-1]
            else:
                anchor, anchor_odom = np.array([[laser_x], [laser_y], [laser_theta]]), odom_now
            self._begin_recovery_session(
                SESSION_MOTION, anchor, anchor_odom=anchor_odom,
                pre_loss_reference=self._recent_reference(),
            )

        # Search around the anchor carried forward by odometry (the robot has moved since)
        search_anchor = self._propagate_pose(self.recovery_anchor, self.recovery_anchor_odom, odom_now)
        jump = self.recovery_kind == SESSION_JUMP
        best, current_w = self.find_best_recovery_pose_anchored(
            ranges, angles, laser_x, laser_y, laser_theta, search_anchor,
            self.recovery_history_snapshot, num_valid_scans,
            include_current=not jump,  # after a jump, AMCL's pose is not a candidate
        )
        best_x, best_y, best_th, best_w, source = best
        required = current_w * self.min_improvement_ratio
        if jump or best_w >= required:
            rospy.loginfo(
                '[LocalizationQuality] POSE_REFINE (%s) inject from %s (current %.3f -> %.3f) '
                'at (%.2f, %.2f, %.2f)',
                self.recovery_kind, source, current_w, best_w, best_x, best_y, best_th,
            )
            self.initial_pose_pub.publish(self._make_initial_pose_msg(
                best_x, best_y, best_th, self.recovery_inject_xy_std, self.recovery_inject_yaw_std, stamp,
            ))
            self._mark_recovery_fired(cooldown=False)  # our own map->odom change is not a jump
        else:
            rospy.loginfo(
                '[LocalizationQuality] POSE_REFINE (%s) keeps AMCL pose (best %.3f %s vs current %.3f, '
                'need %.3f)',
                self.recovery_kind, best_w, source, current_w, required,
            )

        self.pose_refine_done = True
        self.recovery_pose_ready_pub.publish(Bool(data=True))

    def _check_pose(self, mean_score, all_full_range, laser_pose, amcl_fresh_since):
        """
        Shared pass/fail checks. Returns (passed, details).
          - scan match: mean >= floor, and >= ratio x the healthy pre-loss score when known (only for
            scans with the normal range limit: fallback scans have another scale);
          - position / heading close to the anchor carried forward by odometry;
          - AMCL's covariance from an /amcl_pose received since amcl_fresh_since.
        """
        ref = self.pre_loss_reference
        score_ok = mean_score >= self.validation_score_floor
        if ref is not None and all_full_range:
            score_ok = score_ok and mean_score >= self.validation_score_ratio * ref

        if self.recovery_anchor is not None:
            predicted = self._propagate_pose(
                self.recovery_anchor, self.recovery_anchor_odom, self._lookup_odom_pose()
            )
        else:
            predicted = np.array([[laser_pose[0]], [laser_pose[1]], [laser_pose[2]]])
        lx, ly, lth = laser_pose
        dist, yaw_err = self._pose_delta(predicted, lx, ly, lth)
        dist_ok = dist < self.max_acceptance_from_anchor
        yaw_check = self.validation_max_yaw_from_anchor > 0 and self.recovery_anchor_odom is not None
        yaw_ok = (not yaw_check) or yaw_err < self.validation_max_yaw_from_anchor

        amcl = self.last_amcl_pose
        pos_std = yaw_std = float('nan')
        cov_ok = True
        cov_note = 'no fresh /amcl_pose, check skipped'
        if amcl is not None and self.last_amcl_pose_rx >= amcl_fresh_since:
            c = np.array(amcl.pose.covariance).reshape(6, 6)
            pos_std = float(np.sqrt(max(np.linalg.eigvalsh(c[:2, :2]).max(), 0.0)))
            yaw_std = float(np.sqrt(max(c[5, 5], 0.0)))
            cov_ok = pos_std < self.validation_max_position_std and yaw_std < self.validation_max_yaw_std
            cov_note = 'fresh'

        passed = score_ok and dist_ok and yaw_ok and cov_ok
        details = (
            'kind=%s score=%.3f ref=%s%s score_ok=%s | dist=%.2fm dist_ok=%s | yaw_err=%.3f yaw_ok=%s%s | '
            'amcl pos_std=%.3f yaw_std=%.3f cov_ok=%s (%s)'
            % (
                self.recovery_kind, mean_score, 'none' if ref is None else '%.3f' % ref,
                '' if all_full_range else ' (range fallback: ratio unchecked)', score_ok,
                dist, dist_ok, yaw_err, yaw_ok, '' if yaw_check else ' (unchecked)',
                pos_std, yaw_std, cov_ok, cov_note,
            )
        )
        return passed, details

    def _handle_recovery_validate(self, ranges, angles, laser_x, laser_y, laser_theta, num_valid_scans,
                                  used_fallback):
        """Stationary validation: wait for fresh AMCL updates, score a few still scans, publish."""
        if self.validation_done:
            return
        now = rospy.Time.now()
        if self.recovery_anchor is None:
            self.recovery_anchor = np.array([[laser_x], [laser_y], [laser_theta]])
            self.recovery_anchor_odom = self._lookup_odom_pose()

        if self.validation_start_time is None:
            waited = (now - self.validation_enter_time).to_sec()
            if not self._rest_settled() and waited < self.validation_settle_timeout_sec:
                return
            self.validation_start_time = now
            self.validation_scores = []
            self.validation_poses = []
            self.validation_fallback = False
            if not self._rest_settled():
                rospy.logwarn(
                    '[LocalizationQuality] RECOVERY_VALIDATE: stationary AMCL updates not done '
                    'after %.1fs; validating anyway', waited,
                )

        self.validation_scores.append(
            self._score_pose(ranges, angles, laser_x, laser_y, laser_theta, num_valid_scans)
        )
        self.validation_poses.append((laser_x, laser_y, laser_theta))
        self.validation_fallback = self.validation_fallback or used_fallback
        n = self.validation_scan_count
        if len(self.validation_scores) < n:
            return

        passed, details = self._check_pose(
            float(np.mean(self.validation_scores[-n:])), not self.validation_fallback,
            self.validation_poses[-1], self.validation_enter_time,
        )
        self.validation_done = True
        self.recovery_validated_pub.publish(Bool(data=passed))
        if passed:
            rospy.loginfo('[LocalizationQuality] RECOVERY_VALIDATE pass (%s)', details)
            self._end_recovery_session()
        else:
            rospy.logwarn('[LocalizationQuality] RECOVERY_VALIDATE fail (%s)', details)

    def _handle_spin_validate(self, ranges, angles, laser_x, laser_y, laser_theta, num_valid_scans,
                              used_fallback):
        """
        During the relocalize spin: once spin_validation_scans consecutive scans pass every check,
        publish /recovery_validated True so the navigator stops spinning. Never publishes False.
        """
        if not self.recovery_active or self.spin_validated:
            return
        score = self._score_pose(ranges, angles, laser_x, laser_y, laser_theta, num_valid_scans)
        passed, details = self._check_pose(
            score, not used_fallback, (laser_x, laser_y, laser_theta), self.spin_enter_time,
        )
        # Need a fresh AMCL estimate from this spin, not just a fresh-looking score
        fresh = self.last_amcl_pose is not None and self.last_amcl_pose_rx >= self.spin_enter_time
        self.spin_good_scans = self.spin_good_scans + 1 if (passed and fresh) else 0
        if self.spin_good_scans >= self.spin_validation_scans:
            self.spin_validated = True
            spun = (rospy.Time.now() - self.spin_enter_time).to_sec()
            rospy.loginfo(
                '[LocalizationQuality] RELOCALIZING validated after %.1fs of spin (%s)', spun, details,
            )
            self.recovery_validated_pub.publish(Bool(data=True))
            self._end_recovery_session()

    # ------------------------------------------------------------------
    # Idle / park heading correction (fallback)
    # ------------------------------------------------------------------
    def try_heading_correction(self, ranges, angles, laser_x, laser_y, laser_theta, current_weight,
                               num_valid_scans, reason, stamp):
        best_theta, best_weight = self.find_best_heading(
            ranges, angles, laser_x, laser_y, laser_theta, num_valid_scans
        )
        delta = abs(_wrap(best_theta - laser_theta))
        if delta < self.heading_min_delta or best_weight <= current_weight * self.heading_weight_ratio:
            return False
        rospy.loginfo(
            '[LocalizationQuality] %s heading correction: delta=%.3f rad, weight %.4f -> %.4f',
            reason, delta, current_weight, best_weight,
        )
        self.initial_pose_pub.publish(
            self._make_initial_pose_msg(laser_x, laser_y, best_theta, 0.1, np.sqrt(0.05), stamp)
        )
        return True

    # ------------------------------------------------------------------
    # Loss detection (TRACK)
    # ------------------------------------------------------------------
    def _localization_warmup_active(self):
        return (rospy.Time.now() - self.localized_start_time).to_sec() < self.localization_warmup_sec

    def _track_monitor_grace_active(self):
        return (rospy.Time.now() - self.track_monitor_start_time).to_sec() < self.track_monitor_grace_sec

    def _post_recovery_jump_grace_active(self):
        return rospy.Time.now() < self.post_recovery_jump_grace_until

    def _recovery_cooldown_active(self):
        return (rospy.Time.now() - self.last_lost_localization_time).to_sec() < self.recovery_cooldown_sec

    def _mark_recovery_fired(self, cooldown=True):
        now = rospy.Time.now()
        if cooldown:
            self.last_lost_localization_time = now
        self.post_recovery_jump_grace_until = now + rospy.Duration(self.post_recovery_jump_grace_sec)

    def _append_track_history(self, pose, weight, odom_pose):
        self.location_history.append(pose)
        self.odom_history.append(odom_pose)
        self.weight_history.append(weight)
        if len(self.weight_history) > self.max_history_length:
            self.weight_history.pop(0)
        if len(self.location_history) > self.max_location_history_length:
            self.location_history.pop(0)
            self.odom_history.pop(0)

    def _check_jump(self, map_odom, odom_now):
        """
        Size of AMCL's latest map->odom correction, measured at the robot:
        (previous map->odom) (+) odom_now vs (current map->odom) (+) odom_now.
        Returns (delta_xy, delta_theta) or None when there is no previous correction.
        """
        if self._last_map_odom is None or map_odom is None or odom_now is None:
            return None
        before = self._compose(self._last_map_odom, odom_now)
        after = self._compose(map_odom, odom_now)
        return self._pose_delta(before, float(after[0, 0]), float(after[1, 0]), float(after[2, 0]))

    def _trigger_jump_recovery(self, bad_x, bad_y, bad_theta, delta_xy, delta_theta):
        last_pose, last_odom = self.location_history[-1], self.odom_history[-1]
        self._mark_recovery_fired()
        rospy.logwarn(
            'localization_quality: AMCL jump (correction %.2fm / %.3f rad) to (%.2f, %.2f, %.2f); '
            'last good pose (%.2f, %.2f, %.2f)',
            delta_xy, delta_theta, bad_x, bad_y, bad_theta,
            float(last_pose[0, 0]), float(last_pose[1, 0]), float(last_pose[2, 0]),
        )
        self.recovery_fired_pub.publish(Bool(data=True))
        # Jump scans are never appended, so the latest scores are still from before the jump
        self._begin_recovery_session(
            SESSION_JUMP, last_pose, anchor_odom=last_odom, pre_loss_reference=self._recent_reference(),
        )
        self.lost_localization_pub.publish(Bool(data=True))

    def _score_window_means(self):
        """
        Mean of the newest 10 scores vs the middle 10 of the buffer (finite scores only; each window
        needs 5). Returns (None, None) until the buffer is full.
        """
        if len(self.weight_history) < self.max_history_length:
            return None, None
        recent = np.array(self.weight_history[-10:])
        old = np.array(self.weight_history[10:-10])
        recent, old = recent[np.isfinite(recent)], old[np.isfinite(old)]
        if len(recent) < 5 or len(old) < 5:
            return None, None
        return float(np.mean(recent)), float(np.mean(old))

    def _publish_scan_match_diagnostics(self, w_norm, recent_mean=None, old_mean=None):
        self.scan_match_score_pub.publish(Float32(data=float(w_norm)))
        if recent_mean is not None:
            self.scan_match_recent_mean_pub.publish(Float32(data=float(recent_mean)))
        if old_mean is not None:
            self.scan_match_old_mean_pub.publish(Float32(data=float(old_mean)))
        if not self.log_scan_match_diagnostics or self.robot_state != ROBOT_MODE_TRACK:
            return
        now = rospy.Time.now()
        if (now - self._last_diagnostics_log_time).to_sec() < self.diagnostics_log_interval:
            return
        self._last_diagnostics_log_time = now
        parts = ['scan_match_score=%.4f robot_mode=%d' % (w_norm, self.robot_state)]
        if recent_mean is not None and old_mean is not None:
            parts.append('recent_mean=%.4f old_mean=%.4f' % (recent_mean, old_mean))
        rospy.loginfo('localization_quality: %s', ' '.join(parts))

    def _trigger_recovery(self, laser_x, laser_y, laser_theta, recent_mean, old_mean):
        if self._recovery_cooldown_active():
            return
        self._mark_recovery_fired()
        rospy.logwarn(
            'localization_quality: lost localization (recent_mean=%.4f old_mean=%.4f drop_factor=%.2f)',
            recent_mean, old_mean, self.weight_drop_factor,
        )
        self.recovery_fired_pub.publish(Bool(data=True))
        # Anchor on the last pose before the low-score window (the newest 10 scans scored badly);
        # location_history and weight_history are aligned at their ends.
        if len(self.location_history) >= 11:
            anchor, anchor_odom = self.location_history[-11], self.odom_history[-11]
        elif self.location_history:
            anchor, anchor_odom = self.location_history[0], self.odom_history[0]
        else:
            anchor, anchor_odom = np.array([[laser_x], [laser_y], [laser_theta]]), self._lookup_odom_pose()
        # Validate against the healthy score from before the drop, not the degraded recent mean
        self._begin_recovery_session(SESSION_DROP, anchor, anchor_odom=anchor_odom, pre_loss_reference=old_mean)
        self.lost_localization_pub.publish(Bool(data=True))

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def scan_callback(self, msg):
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

        ranges, angles, num_valid_scans, used_fallback = self._preprocess_scan(msg)
        if num_valid_scans == 0:
            return

        # Poses at the scan's own time (the lidar stamps mid-sweep), latest TF if unavailable
        stamp = msg.header.stamp
        laser_pose = self._lookup_pose('/map', '/laser_frame', stamp)
        if laser_pose is None:
            return
        laser_x, laser_y, laser_theta = float(laser_pose[0, 0]), float(laser_pose[1, 0]), float(laser_pose[2, 0])
        current_odom = self._lookup_odom_pose(stamp)

        w = self._score_pose(ranges, angles, laser_x, laser_y, laser_theta, num_valid_scans)

        if self._idle_heading_fix_pending():
            reason = 'park' if self.pending_park_heading_fix else 'idle'
            self.try_heading_correction(
                ranges, angles, laser_x, laser_y, laser_theta, w, num_valid_scans, reason, stamp
            )
            self.made_idle_adjustment = True
            self.pending_park_heading_fix = False
            return

        if self.robot_state == ROBOT_MODE_POSE_REFINE:
            self._handle_pose_refine(ranges, angles, laser_x, laser_y, laser_theta, num_valid_scans, stamp)
            return
        if self.robot_state == ROBOT_MODE_RECOVERY_VALIDATE:
            self._handle_recovery_validate(
                ranges, angles, laser_x, laser_y, laser_theta, num_valid_scans, used_fallback
            )
            return
        if self.robot_state == ROBOT_MODE_RELOCALIZING:
            self._handle_spin_validate(
                ranges, angles, laser_x, laser_y, laser_theta, num_valid_scans, used_fallback
            )
            return
        if self.robot_state != ROBOT_MODE_TRACK:
            self._publish_scan_match_diagnostics(w, None, None)
            return

        # Jump detection: a sudden map->odom correction (independent of how fast the robot moves)
        map_odom = self._lookup_pose('/map', '/odom', stamp)
        jump = self._check_jump(map_odom, current_odom) if self.pose_jump_detection_enabled else None
        if map_odom is not None:
            self._last_map_odom = map_odom
        if jump is not None and (jump[0] > self.max_pose_jump_m or jump[1] > self.max_pose_jump_theta):
            can_recover = (
                bool(self.location_history)
                and not self._localization_warmup_active()
                and not self._track_monitor_grace_active()
                and not self._post_recovery_jump_grace_active()
                and not self._recovery_cooldown_active()
            )
            if can_recover:
                self._trigger_jump_recovery(laser_x, laser_y, laser_theta, jump[0], jump[1])
            else:
                # Accepted: the new map->odom is the reference from now on (no repeated detection)
                rospy.logwarn_throttle(
                    5.0,
                    'localization_quality: AMCL jump (correction %.2fm / %.3f rad) not recovered '
                    '(warmup / grace / cooldown)' % jump,
                )
            return

        self._append_track_history(
            np.array([[laser_x], [laser_y], [laser_theta]]), float('nan') if used_fallback else w, current_odom,
        )
        recent_mean, old_mean = self._score_window_means()
        self._publish_scan_match_diagnostics(w, recent_mean, old_mean)

        if (
            recent_mean is None
            or self._localization_warmup_active()
            or self._track_monitor_grace_active()
        ):
            return
        score_drop = recent_mean < self.weight_drop_factor * old_mean
        score_floor = self.use_absolute_score_floor and recent_mean < self.mean_weight_limit
        if score_drop or score_floor:
            self._trigger_recovery(laser_x, laser_y, laser_theta, recent_mean, old_mean)

    def robot_state_callback(self, msg):
        with self._lock:
            self._robot_state_callback_locked(msg)

    def _robot_state_callback_locked(self, msg):
        prev_state = self.robot_state
        new_state = msg.data
        now = rospy.Time.now()
        if new_state == ROBOT_MODE_IDLE and prev_state != ROBOT_MODE_IDLE:
            self.idle_enter_time = now
            self.made_idle_adjustment = False
            if prev_state in RECOVERY_MODES and self.recovery_active:
                self._end_recovery_session()  # the navigator gave up or finished
        if prev_state == ROBOT_MODE_PARK_HEADING and new_state == ROBOT_MODE_IDLE:
            self.pending_park_heading_fix = True
            self.park_fix_ready_time = now + rospy.Duration(self.park_settle_duration)
            rospy.loginfo(
                '[LocalizationQuality] Park complete; heading correction scheduled in %.1fs',
                self.park_settle_duration,
            )
        if new_state == ROBOT_MODE_POSE_REFINE and prev_state != ROBOT_MODE_POSE_REFINE:
            self.pose_refine_enter_time = now
            self.pose_refine_done = False
            rospy.loginfo(
                '[LocalizationQuality] POSE_REFINE entered (%s); search once still (after %.1fs)',
                self.recovery_kind or SESSION_MOTION, self.post_backing_settle_sec,
            )
        if new_state == ROBOT_MODE_RECOVERY_VALIDATE and prev_state != ROBOT_MODE_RECOVERY_VALIDATE:
            self.validation_enter_time = now
            self.validation_start_time = None
            self.validation_scores = []
            self.validation_poses = []
            self.validation_done = False
            self._request_rest_batch()  # AMCL updates after any pose injected in POSE_REFINE
            rospy.loginfo(
                '[LocalizationQuality] RECOVERY_VALIDATE entered; stationary AMCL updates (max %.1fs), '
                'then %d scans', self.validation_settle_timeout_sec, self.validation_scan_count,
            )
        if new_state == ROBOT_MODE_RELOCALIZING and prev_state != ROBOT_MODE_RELOCALIZING:
            self.spin_enter_time = now
            self.spin_good_scans = 0
            self.spin_validated = False
            rospy.loginfo(
                '[LocalizationQuality] RELOCALIZING entered; validating each scan (%d in a row ends the spin)',
                self.spin_validation_scans,
            )
        if prev_state != ROBOT_MODE_TRACK and new_state == ROBOT_MODE_TRACK:
            self.track_monitor_start_time = now
            self.weight_history = []
            self.location_history = []
            self.odom_history = []
            self._last_map_odom = None
            if self.log_scan_match_diagnostics:
                rospy.loginfo(
                    '[LocalizationQuality] TRACK started; score history cleared, %.1fs monitor grace',
                    self.track_monitor_grace_sec,
                )
        self.robot_state = new_state

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
