# Monte Carlo Localization

This package implements monte carlo (particle filter) localization (MCL) for the robot. This algorithm is based on "[Probabilistic Robotics](https://mitpress.mit.edu/9780262201629/probabilistic-robotics/)" by Thrun, Burgard, and Fox (2005).

## Scripts
- **mc_localization.py**: Implements the vanilla MCL.
- **mc_localization_augmented.py**: Implements Monte Carlo Localization with an adaptive number of particles (not fully implemented)
- **mc_localization_kld.py**: Implements Monte Carlo Localization with particle sampling based on KL Divergence (not fully implemented)
- **map_loader.py**: Loads a map and publishes it to the map topic

## Map Generation
To use a map for localization, follow these instructions:

1) Save the map using map_server: `rosrun map_server map_saver [--occ <threshold_occupied>] [--free <threshold_free>] [-f <mapname>] map:=/your/costmap/topic`
2) My code requires a second map where the user indicates "off limits" locations. To create this map, upload the .pgm file into GIMP, draw in black anywhere you want to add a new boundary. Save as a new .pgm file with the same map name with with '_mod' appended to the file name.
3) Move the map files (`<map_name_prefix>.pgm`, `<map_name_prefix>_mod.pgm`, `<map_name_prefix>.yaml`) into the `./maps` directory. 
4) Create an occupancy grid by running `python3 ./generate_dds_map.py --map_file <map_name_prefix>` within the `./scripts` directory.

Now, the occupancy grid for this map will be saved as `./lookup_table/current_map.npy` and can be easily loaded by the MCL script. Additionally, the map is saved as a json in the `./map_json` directory.

If you have an occupancy grid map you want to add, simply use the same name convention but with a '_occ' appended to the .pgm file name.

## Launch
After creating a 
- **mcl.launch**: Launches the MCL localization file `mc_localization.py`

#### Arguments
- **num_particles** (default: 200): Number of particles used in the filter
- **lidar_measurement_skip** (default: 2): Number of lidar measurements to skip
- **z_hit** (default: 0.75): Weight for measurements that hit the expected obstacle
- **z_random** (default: 0.25): Weight for random measurements
- **sigma_hit** (default: 0.01): Standard deviation for hit measurements
- **alpha1** (default: 0.02): Motion model noise parameter
- **alpha2** (default: 0.1): Motion model noise parameter
- **alpha3** (default: 0.2): Motion model noise parameter
- **alpha4** (default: 0.02): Motion model noise parameter

### Running the launch file:
```
roslaunch mattbot_mcl mcl.launch
```

To include an argument, modify the launch statement as follows:
```
roslaunch mattbot_mcl mcl.launch num_particles:=250 alpha1:=0.01
```



**Author**: Matthew Sato, Engineering Informatics Lab, Stanford University

**License**: This package is released under the [MIT license](LICENSE).