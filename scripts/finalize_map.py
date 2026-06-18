#!/usr/bin/env python3
"""Save a SLAM map to mattbot_mcl/maps and run DDS/MCL preparation."""

import argparse
import os
import subprocess
import sys

import rospkg

_PKG_PATH = rospkg.RosPack().get_path("mattbot_mcl")
_SCRIPTS_DIR = os.path.join(_PKG_PATH, "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from generate_dds_map import MAP_JSON_DIR, LOOKUP_TABLE_DIR, MAPS_DIR, MapPreparer


def main():
    parser = argparse.ArgumentParser(description="Save map and prepare DDS/MCL artifacts")
    parser.add_argument("--name", required=True, help="Map basename (no extension)")
    parser.add_argument("--skip-save", action="store_true", help="Skip map_saver (PGM/YAML already written)")
    parser.add_argument("--map-topic", default="/map", help="OccupancyGrid topic for map_saver")
    args = parser.parse_args()

    base_path = os.path.join(MAPS_DIR, args.name)

    if not args.skip_save:
        print(f"Saving map to {base_path}...")
        subprocess.check_call(
            ["rosrun", "map_server", "map_saver", "-f", base_path, f"map:={args.map_topic}"]
        )

    if not os.path.isfile(base_path + ".pgm") or not os.path.isfile(base_path + ".yaml"):
        sys.exit(f"Map files not found at {base_path}.{{pgm,yaml}}")

    print(f"Preparing DDS map for '{args.name}'...")
    MapPreparer(args.name, auto_mod=True, show_plot=False)

    print("Done.")
    print(f"  maps:         {base_path}.pgm, {base_path}.yaml")
    print(f"  lookup_table: {os.path.join(LOOKUP_TABLE_DIR, 'current_map.npy')}")
    print(f"  map_json:     {os.path.join(MAP_JSON_DIR, 'current_map.json')}")


if __name__ == "__main__":
    main()
