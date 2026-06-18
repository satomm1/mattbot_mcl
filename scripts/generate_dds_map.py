from nav_msgs.msg import OccupancyGrid, MapMetaData
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import argparse
import json
import sys

import rospkg
from tf.transformations import euler_from_quaternion

from utils.grids import StochOccupancyGrid2D, DetOccupancyGrid2D

_PKG_PATH = rospkg.RosPack().get_path("mattbot_mcl")
MAPS_DIR = os.path.join(_PKG_PATH, "maps")
LOOKUP_TABLE_DIR = os.path.join(_PKG_PATH, "lookup_table")
MAP_JSON_DIR = os.path.join(_PKG_PATH, "map_json")


def write_occupancy_grid_to_ros_map(msg, base_path):
    """Write OccupancyGrid to map_server-style .pgm + .yaml at base_path (no extension)."""
    info = msg.info
    w, h = info.width, info.height
    grid = np.array(msg.data, dtype=np.int8).reshape((h, w))
    img = np.full((h, w), 205, dtype=np.uint8)
    img[grid == 0] = 254
    img[grid >= 50] = 0
    img = np.flipud(img)

    pgm_path = base_path + ".pgm"
    with open(pgm_path, "wb") as f:
        f.write(b"P5\n")
        f.write(f"# CREATOR: finalize_map.py {info.resolution:.3f} m/pix\n".encode())
        f.write(f"{w} {h}\n255\n".encode())
        f.write(img.tobytes())

    q = info.origin.orientation
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    yaml_path = base_path + ".yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"image: {os.path.basename(pgm_path)}\n")
        f.write(f"resolution: {info.resolution}\n")
        f.write(f"origin: [{info.origin.position.x}, {info.origin.position.y}, {yaw}]\n")
        f.write("negate: 0\n")
        f.write("occupied_thresh: 0.65\n")
        f.write("free_thresh: 0.196\n")
    return pgm_path, yaml_path


def complete_map_names(maps_dir):
    """
    Map names that have {name}.yaml and {name}.pgm in maps_dir.
    {name}_mod.pgm is optional; it can be created when running this script.
    """
    if not os.path.isdir(maps_dir):
        return []
    names = []
    for entry in os.listdir(maps_dir):
        if not entry.endswith(".yaml"):
            continue
        name = entry[: -len(".yaml")]
        if not name:
            continue
        base = os.path.join(maps_dir, name)
        if os.path.isfile(base + ".yaml") and os.path.isfile(base + ".pgm"):
            names.append(name)
    return sorted(set(names))


def maps_help_epilog(maps_dir):
    names = complete_map_names(maps_dir)
    lines = [
        "Available --map_file names (require <name>.yaml and <name>.pgm under mattbot_mcl/maps; <name>_mod.pgm optional):",
    ]
    if not os.path.isdir(maps_dir):
        lines.append(f"  (maps directory not found: {maps_dir})")
    elif not names:
        lines.append("  (no complete map sets found)")
    else:
        lines.extend(f"  {n}" for n in names)
    return "\n".join(lines)

class MapPreparer:
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

    def __init__(self, map_file, *, auto_mod=False, show_plot=True):
        """
        Initializes the MapLoader class

        Args:
            map_file: The file containing the map information in yaml format
            auto_mod: Copy base .pgm to _mod.pgm when mod file is missing
            show_plot: Show lookup-table plot after generation
        """
        self.map_seq = 0
        self.auto_mod = auto_mod
        self.show_plot = show_plot

        self.map_data, self.map_metadata = self.load_map(map_file)
        self.resolution = self.map_metadata.resolution

        lookup_path = os.path.join(LOOKUP_TABLE_DIR, map_file + ".npy")
        current_lookup = os.path.join(LOOKUP_TABLE_DIR, "current_map")
        if os.path.exists(lookup_path):
            print("Loading Lookup Table...")
            self.dist_lookup_table = np.load(lookup_path)
            np.save(current_lookup, self.dist_lookup_table)
            print("Lookup Table Loaded")
        else:
            print("Generating Lookup Table...")
            self.dist_lookup_table = self.generate_dist_lookup_table()
            self.dist_lookup_table = self.dist_lookup_table.T
            print("Lookup Table Generated")
            np.save(lookup_path, self.dist_lookup_table)
            np.save(current_lookup, self.dist_lookup_table)

            if self.show_plot:
                fig, ax = plt.subplots()
                cbar = ax.imshow(self.dist_lookup_table, cmap="hot")
                fig.colorbar(cbar)
                plt.show()

    def load_map(self, map_file):
        """
        Loads the map from the specified file

        Args:
            map_file: The file containing the map

        Returns:
            The map as a 2D occupancy grid
        """
        with open(os.path.join(MAPS_DIR, map_file + ".yaml"), "r") as f:
            map_data = yaml.safe_load(f)

        pgm_file = map_data['image']
        pgm_mod_file = pgm_file.split('.')[0] + '_mod.pgm'
        pgm_occ_file = pgm_file.split('.')[0] + '_occ.pgm'
        resolution = map_data['resolution']
        origin = map_data['origin']

        base_pgm_path = os.path.join(MAPS_DIR, pgm_file)
        mod_pgm_path = os.path.join(MAPS_DIR, pgm_mod_file)

        if not os.path.isfile(base_pgm_path):
            raise FileNotFoundError(f"Base map image not found: {base_pgm_path}")

        if not os.path.isfile(mod_pgm_path):
            if self.auto_mod:
                shutil.copy2(base_pgm_path, mod_pgm_path)
                print(f"Copied {base_pgm_path} -> {mod_pgm_path}")
            else:
                print(f"No modified map file found: {mod_pgm_path}")
                while True:
                    ans = input("Use the original map as the modified map? [y/n]: ").strip().lower()
                    if ans in ("y", "yes"):
                        shutil.copy2(base_pgm_path, mod_pgm_path)
                        print(f"Copied {base_pgm_path} -> {mod_pgm_path}")
                        break
                    if ans in ("n", "no"):
                        raise SystemExit("Aborted: add a _mod.pgm map or run again and answer y.")
                    print("Please answer y or n.")

        with open(base_pgm_path, 'rb') as f:
            pgm_data = plt.imread(f)

        with open(mod_pgm_path, 'rb') as f:
            pgm_data_mod = plt.imread(f)


        # find range of values in pgm_data where value is not 205
        occupied_loc = np.where(pgm_data != 205)
        min_x = np.min(occupied_loc[0])
        max_x = np.max(occupied_loc[0])
        min_y = np.min(occupied_loc[1])
        max_y = np.max(occupied_loc[1])

        # Only get areas of map we care about
        map = pgm_data[min_x:max_x + 1, min_y:max_y + 1]
        map = np.array(map).astype(int)

        mod_map = pgm_data_mod[min_x:max_x + 1, min_y:max_y + 1]
        mod_map = np.array(mod_map).astype(int)

        # Convert to occupancy grid values
        unique_map_values = np.unique(map)
        print("Unique map values:", unique_map_values)
        for unique_value in unique_map_values:
            if unique_value not in [205, 254, 0]:
                # Map to the closest value among 205, 254, and 0
                closest_value = min([205, 254, 0], key=lambda x: abs(x - unique_value))
                map[map == unique_value] = closest_value

        unknown_loc = np.where(map == 205)
        free_loc = np.where(map == 254)
        occupied_loc = np.where(map == 0)
        map[unknown_loc] = -1
        map[free_loc] = 0
        map[occupied_loc] = 100
         

        unique_map_mod_values = np.unique(mod_map)
        print("Unique mod map values:", unique_map_mod_values)
        # Find where the map_mod is not 254 and not 205
        mod_loc = np.where((mod_map != 254) & (mod_map != 205))
        mod_map = map.copy()
        mod_map[mod_loc] = 100
        

        if os.path.exists(os.path.join(MAPS_DIR, pgm_occ_file)):
            with open(os.path.join(MAPS_DIR, pgm_occ_file), 'rb') as f:
                pgm_data_occ = plt.imread(f)

            occ_map = np.array(pgm_data_occ).astype(int)
            occ_map = np.flip(occ_map, 1)
            occ_map = np.flip(occ_map, 0)

            occ_unknown_loc = np.where(occ_map == 205)
            occ_free_loc = np.where(occ_map == 254)
            occ_occupied_loc = np.where(occ_map == 0)
            mod_map[occ_occupied_loc] = 100

        map = np.flip(map, 1)
        mod_map = np.flip(mod_map, 1)

        flattened_map = map.flatten(order='C')  # Flatten to row-major order
        flattened_mod_map = mod_map.flatten(order='C')

        md_msg = MapMetaData()
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

        map_dict = dict()
        map_dict['map'] = dict()
        map_dict['map']['width'] = md_msg.width
        map_dict['map']['height'] = md_msg.height
        map_dict['map']['origin_x'] = md_msg.origin.position.x
        map_dict['map']['origin_y'] = md_msg.origin.position.y
        map_dict['map']['origin_z'] = md_msg.origin.position.z
        map_dict['map']['origin_orientation_x'] = md_msg.origin.orientation.x
        map_dict['map']['origin_orientation_y'] = md_msg.origin.orientation.y
        map_dict['map']['origin_orientation_z'] = md_msg.origin.orientation.z
        map_dict['map']['origin_orientation_w'] = md_msg.origin.orientation.w
        map_dict['map']['resolution'] = md_msg.resolution
        map_dict['map']['occupancy'] = flattened_map.tolist()

        map_mod_dict = dict()
        map_mod_dict['map'] = dict()
        map_mod_dict['map']['width'] = md_msg.width
        map_mod_dict['map']['height'] = md_msg.height
        map_mod_dict['map']['origin_x'] = md_msg.origin.position.x
        map_mod_dict['map']['origin_y'] = md_msg.origin.position.y
        map_mod_dict['map']['origin_z'] = md_msg.origin.position.z
        map_mod_dict['map']['origin_orientation_x'] = md_msg.origin.orientation.x
        map_mod_dict['map']['origin_orientation_y'] = md_msg.origin.orientation.y
        map_mod_dict['map']['origin_orientation_z'] = md_msg.origin.orientation.z
        map_mod_dict['map']['origin_orientation_w'] = md_msg.origin.orientation.w
        map_mod_dict['map']['resolution'] = md_msg.resolution
        map_mod_dict['map']['occupancy'] = flattened_mod_map.tolist()

        data_dict = dict()
        data_dict['data'] = map_dict
        mod_data_dict = dict()
        mod_data_dict['data'] = map_mod_dict

        map_json = json.dumps(data_dict)
        with open(os.path.join(MAP_JSON_DIR, map_file + ".json"), "w") as f:
            f.write(map_json)

        with open(os.path.join(MAP_JSON_DIR, "current_map.json"), "w") as f:
            f.write(map_json)

        with open(os.path.join(MAP_JSON_DIR, "current_map_mod.json"), "w") as f:
            f.write(json.dumps(mod_data_dict))

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



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Generate DDS Map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=maps_help_epilog(MAPS_DIR),
    )
    parser.add_argument(
        "--map_file",
        type=str,
        metavar="NAME",
        help="Map basename (no extension): loads mattbot_mcl/maps/<NAME>.yaml and <NAME>.pgm; <NAME>_mod.pgm optional (prompt to copy from .pgm if missing)",
        default="map_aligned",
    )
    parser.add_argument(
        "--auto-mod",
        action="store_true",
        help="Copy base .pgm to _mod.pgm when mod file is missing (no prompt)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip lookup-table plot after generation",
    )
    args = parser.parse_args()
    map_file = args.map_file
    show_plot = sys.stdout.isatty() and not args.no_plot

    map_loader = MapPreparer(map_file, auto_mod=args.auto_mod, show_plot=show_plot)
