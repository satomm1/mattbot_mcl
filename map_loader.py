import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData
import yaml
import matplotlib.pyplot as plt
import numpy as np

class MapLoader:
    """
    The MapLoader class

    Attributes:
        TODO Add me

    Description:
        The MapLoader class is responsible for loading the map from the specified file and publishing it to the /map
        topic. The map is represented as a 2D occupancy grid. The map is published periodically at a rate of 1 Hz.
    """

    def __init__(self, map_file):
        """
        Initializes the MapLoader class

        Args:
            map_file: The file containing the map information in yaml format
        """
        rospy.init_node('map_loader')

        # Map and map metadata publisher
        self.map_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=10)
        self.map_md_pub = rospy.Publisher('/map_metadata', MapMetaData, queue_size=10)

        self.map_seq = 0

        self.map_data, self.map_metadata = self.load_map(map_file)

    def load_map(self, map_file):
        """
        Loads the map from the specified file

        Args:
            map_file: The file containing the map

        Returns:
            The map as a 2D occupancy grid
        """
        with open(map_file, 'r') as f:
            map_data = yaml.safe_load(f)

        pgm_file = map_data['image']
        resolution = map_data['resolution']
        origin = map_data['origin']

        with open('maps/' + pgm_file, 'rb') as f:
            pgm_data = plt.imread(f)

        # find range of values in pgm_data where value is not 205
        occupied_loc = np.where(pgm_data != 205)
        min_x = np.min(occupied_loc[0])
        max_x = np.max(occupied_loc[0])
        min_y = np.min(occupied_loc[1])
        max_y = np.max(occupied_loc[1])

        # Only get areas of map we care about
        map = pgm_data[min_x:max_x, min_y:max_y]
        map = np.array(map).astype(int)

        # Convert to occupancy grid values
        unknown_loc = np.where(map == 205)
        free_loc = np.where(map == 254)
        occupied_loc = np.where(map == 0)
        map[unknown_loc] = -1
        map[free_loc] = 0
        map[occupied_loc] = 100

        flattened_map = map.flatten(order='C')  # Flatten to row-major order

        md_msg = MapMetaData()
        md_msg.map_load_time = rospy.Time.now()
        md_msg.resolution = resolution
        md_msg.width = map.shape[1]
        md_msg.height = map.shape[0]
        md_msg.origin.position.x = 0
        md_msg.origin.position.y = 0
        md_msg.origin.position.z = 0
        md_msg.origin.orientation.x = 0
        md_msg.origin.orientation.y = 0
        md_msg.origin.orientation.z = 0
        md_msg.origin.orientation.w = 1

        return flattened_map, md_msg

    def publish_map(self, map_data, map_metadata):
        """
        Publishes the map to the /map and /map_metadata topics
        """
        map_msg = OccupancyGrid()
        map_msg.header.stamp = rospy.Time.now()
        map_msg.header.frame_id = 'map'
        map_msg.header.seq = self.map_seq
        self.map_seq += 1

        map_msg.info = map_metadata
        map_msg.data = map_data

        self.map_pub.publish(map_msg)
        self.map_md_pub.publish(map_metadata)

    def run(self):
        """
        Runs the map loader node
        """
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.publish_map(self.map_data, self.map_metadata)
            rate.sleep()
        pass

if __name__ == '__main__':
    map_file = 'maps/map.yaml'
    map_loader = MapLoader(map_file)
    map_loader.run()