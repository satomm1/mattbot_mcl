import numpy as np
import yaml
import matplotlib.pyplot as plt
import os
from utils.grids import StochOccupancyGrid2D
from tqdm import tqdm

def load_map(map_file):
    """
    Loads the map from the specified file

    Args:
        map_file: The file containing the map

    Returns:
        The map as a 2D occupancy grid
    """
    with open('maps/' + map_file + '.yaml', 'r') as f:
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

    map = np.flip(map, 1)

    flattened_map = map.flatten(order='C')  # Flatten to row-major order

    width = map.shape[1]
    height = map.shape[0]


    return flattened_map, resolution, width, height

def generate_dist_lookup_table(map_width, map_height, map_resolution, map_data, map_originx=0, map_originy=0):
    """
    Generates a table of distances from the LIDAR sensor to the nearest occupied cell
    """
    occupancy = StochOccupancyGrid2D(
        map_resolution,
        map_width,
        map_height,
        map_originx,
        map_originy,
        3,
        map_data,
    )

    # Get the indices of every cell in the map that is occupied
    probs = np.reshape(np.asarray(map_data), (height, width))
    occupied_indices = np.where(probs == 100)
    occupied_indices = np.vstack(occupied_indices)

    lookup_table = np.zeros((map_width, map_height))
    # use tqdm to show progress bar
    for x in tqdm(range(map_width)):
        for y in range(map_height):
            if occupancy.is_unknown((x*map_resolution,y*map_resolution)):
                lookup_table[x,y] = -1
            elif occupancy.is_free((x*map_resolution,y*map_resolution)):
                # lookup_table[x, y] = find_closest_obstacle(x, y, map_width, map_height, map_resolution, occupancy)
                lookup_table[x, y] = np.min(np.linalg.norm(np.array([y, x]) - occupied_indices.T, axis=1)) * resolution
            else:
                lookup_table[x, y] = 0


    return lookup_table

def find_closest_obstacle(x, y, map_width, map_height, map_resolution, occupancy):
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
                    new_x = np.clip(x+i, 0, map_width-1)
                    new_y = np.clip(y+j, 0, map_height-1)
                    if ~occupancy.is_free((new_x*map_resolution, new_y*map_resolution)):
                        return np.sqrt(i**2 + j**2)*map_resolution
        k += 1


map_file = "map_aligned"
map_data, resolution, width, height = load_map(map_file)

# if os.path.exists('lookup_table/' + map_file + '.npy'):
#     print("Loading Lookup Table...")
#     dist_lookup_table = np.load('lookup_table/' + map_file + '.npy')
#     # np.save('lookup_table/mattbot_map', dist_lookup_table)
#     print("Lookup Table Loaded")
#     fig, ax = plt.subplots()
#     cbar = ax.imshow(dist_lookup_table, cmap='hot')
#     fig.colorbar(cbar)
#     plt.show()
# else:
print("Generating Lookup Table...")
dist_lookup_table = generate_dist_lookup_table(width, height, resolution, map_data)
dist_lookup_table = dist_lookup_table.T
print("Lookup Table Generated")
# np.save('lookup_table/' + map_file, dist_lookup_table)
#
# np.save('lookup_table/mattbot_map', dist_lookup_table)

fig, ax = plt.subplots()
cbar = ax.imshow(dist_lookup_table, cmap='hot')
fig.colorbar(cbar)
plt.show()
