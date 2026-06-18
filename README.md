# Monte Carlo Localization

This package implements monte carlo (particle filter) localization (MCL) for the robot. This algorithm is based on "[Probabilistic Robotics](https://mitpress.mit.edu/9780262201629/probabilistic-robotics/)" by Thrun, Burgard, and Fox (2005).

## Scripts
- **mc_localization.py**: Implements the vanilla MCL.
- **mc_localization_augmented.py**: Implements Monte Carlo Localization with an adaptive number of particles (not fully implemented)
- **mc_localization_kld.py**: Implements Monte Carlo Localization with particle sampling based on KL Divergence (not fully implemented)
- **map_loader.py**: Loads a map and publishes it to the map topic

> [!IMPORTANT]
> Right now, you must edit the `init_particles()` function in `mc_localization.py` to set coordinates close to the starting point of the mobile robot. Eventually, we will automate this process to make this easier.

## Map Generation
To use a map for localization:

**While mapping** (`sense_and_map.launch` running):
```bash
rosrun mattbot_mcl finalize_map.py --name <map_name_prefix>
```
This saves `<map_name_prefix>.pgm`/`.yaml` into `./maps`, copies a `_mod.pgm` if missing, and generates the DDS lookup table and JSON.

**Auto-save on exit:** `sense_and_map.launch` saves to `autosave` when you Ctrl+C (enabled by default). Disable with `auto_save:=false`, or use a different name with `auto_save_map_name:=scratch`. The `autosave` files are overwritten each session.

**Optional no-go zones:** Edit `<map_name_prefix>_mod.pgm` in GIMP (draw black where the robot should not go), then re-run:
```bash
rosrun mattbot_mcl finalize_map.py --name <map_name_prefix> --skip-save
```
Or use the lower-level script directly:
```bash
python3 ./generate_dds_map.py --map_file <map_name_prefix> --auto-mod --no-plot
```

**Optional supplemental occupancy:** Add `<map_name_prefix>_occ.pgm` (same naming convention as before).

Artifacts: `./lookup_table/current_map.npy` (MCL) and `./map_json/current_map.json` / `current_map_mod.json` (DDS).

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