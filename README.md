# Monte Carlo Localization

This package implements monte carlo (particle filter) localization (MCL) for the robot. This algorithm is based on "[Probabilistic Robotics](https://mitpress.mit.edu/9780262201629/probabilistic-robotics/)" by Thrun, Burgard, and Fox (2005).

### Scripts
- **mc_localization.py**: Implements the vanilla MCL.
- **mc_localization_augmented.py**: Implements Monte Carlo Localization with an adaptive number of particles (not fully implemented)
- **mc_localization_kld.py**: Implements Monte Carlo Localization with particle sampling based on KL Divergence (not fully implemented)
- **map_loader.py**: Loads a map and publishes it to the map topic

### Launch
- **mcl.launch**: Launches the MCL localization file `mc_localization.py`

### Map Generation
If you have a map that you want to use, follow these instructions:

1) Save the map using map_server: `rosrun map_server map_saver [--occ <threshold_occupied>] [--free <threshold_free>] [-f <mapname>] map:=/your/costmap/topic`
2) My code requires a second map where the user indicates "off limits" locations. To create this map, upload the .pgm file into GIMP, draw in black anywhere you want to add a new boundary. Save as a new .pgm file with the same map name with with '_mod' appended to the file name.
3) Create an occupancy grid by running `python3 ./generate_dds_map.py --map_file <map_name_prefix>` within the `./scripts` directory.

If you have an occupancy grid map you want to add, simply use the same name convention but with a '_occ' appended to the .pgm file name.

**Author**: Matthew Sato, Engineering Informatics Lab, Stanford University

**License**: This package is released under the [MIT license](LICENSE).