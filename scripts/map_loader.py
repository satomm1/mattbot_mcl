import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os

from utils.grids import StochOccupancyGrid2D, DetOccupancyGrid2D

class MapPublisher:
    """
    The MapLoader class

    Attributes:
        map_pub: The map publisher
        map_md_pub: The map metadata publisher
        map_seq: The sequence number for the map message
        map_data: The map data (2D occupancy grid data)
        map_metadata: The map metadata (MapMetaData message)
        resolution: The resolution of the map (meters per pixel)

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
        self.resolution = self.map_metadata.resolution

        # if os.path.exists('lookup_table/' + map_file + '.npy'):
        #     print("Loading Lookup Table...")
        #     self.dist_lookup_table = np.load('lookup_table/' + map_file + '.npy')
        #     np.save('lookup_table/mattbot_map', self.dist_lookup_table)
        #     print("Lookup Table Loaded")
        #     # fig, ax = plt.subplots()
        #     # cbar = ax.imshow(self.dist_lookup_table, cmap='hot')
        #     # fig.colorbar(cbar)
        #     # plt.show()
        # else:
        #     print("Generating Lookup Table...")
        #     self.dist_lookup_table =self.generate_dist_lookup_table()
        #     self.dist_lookup_table = self.dist_lookup_table.T
        #     print("Lookup Table Generated")
        #     np.save('lookup_table/' + map_file, self.dist_lookup_table)

        #     np.save('lookup_table/mattbot_map', self.dist_lookup_table)

        #     fig, ax = plt.subplots()
        #     cbar = ax.imshow(self.dist_lookup_table, cmap='hot')
        #     fig.colorbar(cbar)
        #     plt.show()

    def load_map(self, map_file):
        """
        Loads the map from the specified file

        Args:
            map_file: The file containing the map

        Returns:
            The map as a 2D occupancy grid
        """
        with open('../maps/' + map_file + '.yaml', 'r') as f:
            map_data = yaml.safe_load(f)

        pgm_file = map_data['image']
        resolution = map_data['resolution']
        origin = map_data['origin']

        with open('../maps/' + pgm_file, 'rb') as f:
            pgm_data = plt.imread(f)

        # Plot locations with pgm_data == 204
        indx = np.where(pgm_data == 204)
        plt.scatter(indx[1], indx[0], c='r', s=1)
        plt.savefig("test3.png")


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
        mod_loc = np.where(map == 204)
        map[unknown_loc] = -1
        map[free_loc] = 0
        map[occupied_loc] = 100
        map[mod_loc] = 100
        
        map = np.flip(map, 1)

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

    def generate_dist_lookup_table(self):
        """
        Generates a table of distances from the LIDAR sensor to the nearest occupied cell
        """
        self.map_width = self.map_metadata.width
        self.map_height = self.map_metadata.height
        self.map_resolution = self.map_metadata.resolution
        self.map_originx = self.map_metadata.origin.position.x
        self.map_originy = self.map_metadata.origin.position.y

        self.occupancy = StochOccupancyGrid2D(
            self.map_resolution,
            self.map_width,
            self.map_height,
            self.map_originx,
            self.map_originy,
            3,
            self.map_data,
        )

        # Get the indices of every cell in the map that is occupied
        probs = np.reshape(np.asarray(self.map_data), (self.map_height, self.map_width))
        occupied_indices = np.where(probs == 100)
        occupied_indices = np.vstack(occupied_indices)

        lookup_table = np.zeros((self.map_width, self.map_height))
        for x in range(self.map_width):
            for y in range(self.map_height):
                if self.occupancy.is_unknown((x*self.map_resolution,y*self.map_resolution)):
                    lookup_table[x,y] = -1
                elif self.occupancy.is_free((x*self.map_resolution,y*self.map_resolution)):
                    # lookup_table[x, y] = self.find_closest_obstacle(x, y)
                    lookup_table[x, y] = np.min(np.linalg.norm(np.array([y, x]) - occupied_indices.T, axis=1)) * self.map_resolution
                else:
                    lookup_table[x, y] = 0
        return lookup_table

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
                        if ~self.occupancy.is_free((new_x*self.map_resolution, new_y*self.map_resolution)):
                            return np.sqrt(i**2 + j**2)*self.map_resolution
            k += 1

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

        self.map_md_pub.publish(map_metadata)
        self.map_pub.publish(map_msg)


    def run(self):
        """
        Runs the map loader node
        """
        rate = rospy.Rate(0.5)  # 0.5 Hz
        while not rospy.is_shutdown():
            self.publish_map(self.map_data, self.map_metadata)
            rate.sleep()
        pass

if __name__ == '__main__':
    map_file = 'fullmap5_mod'
    map_loader = MapPublisher(map_file)
    map_loader.run()
