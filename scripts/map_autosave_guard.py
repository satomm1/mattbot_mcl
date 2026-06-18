#!/usr/bin/env python3
"""Cache /map during SLAM and auto-finalize on roslaunch shutdown."""

import os
import sys

import rospy
import rospkg
from nav_msgs.msg import OccupancyGrid

_PKG_PATH = rospkg.RosPack().get_path("mattbot_mcl")
_SCRIPTS_DIR = os.path.join(_PKG_PATH, "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from generate_dds_map import MAPS_DIR, MapPreparer, write_occupancy_grid_to_ros_map


class MapAutosaveGuard:
    def __init__(self):
        self.auto_save = rospy.get_param("~auto_save", True)
        self.map_name = rospy.get_param("~map_name", "autosave")
        self.map_topic = rospy.get_param("~map_topic", "/map")
        self._cached_map = None
        rospy.Subscriber(self.map_topic, OccupancyGrid, self._map_cb, queue_size=1)
        rospy.on_shutdown(self._on_shutdown)

    def _map_cb(self, msg):
        self._cached_map = msg

    def _on_shutdown(self):
        if not self.auto_save:
            return
        if self._cached_map is None:
            rospy.logwarn("map_autosave_guard: no map received; skipping auto-save")
            return

        rospy.loginfo("map_autosave_guard: auto-saving map '%s'...", self.map_name)
        base_path = os.path.join(MAPS_DIR, self.map_name)
        write_occupancy_grid_to_ros_map(self._cached_map, base_path)
        MapPreparer(self.map_name, auto_mod=True, show_plot=False)
        rospy.loginfo("map_autosave_guard: saved '%s' to %s", self.map_name, MAPS_DIR)


if __name__ == "__main__":
    rospy.init_node("map_autosave_guard")
    MapAutosaveGuard()
    rospy.spin()
