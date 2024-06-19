# Monte Carlo Localization

This package implements monte carlo (particle filter) localization (MCL) for the robot. This algorithm is based on "[Probabilistic Robotics](https://mitpress.mit.edu/9780262201629/probabilistic-robotics/)" by Thrun, Burgard, and Fox (2005).

### Scripts
- **mc_localization.py**: Implements the vanilla MCL.
- **mc_localization_augmented.py**: Implements Monte Carlo Localization with an adaptive number of particles (not fully implemented)
- **mc_localization_kld.py**: Implements Monte Carlo Localization with particle sampling based on KL Divergence (not fully implemented)
- **map_loader.py**: Loads a map and publishes it to the map topic

### Launch
- **mcl.launch**: Launches the MCL localization file `mc_localization.py`

**Author**: Matthew Sato, Engineering Informatics Lab, Stanford University

**License**: This package is released under the [MIT license](LICENSE).