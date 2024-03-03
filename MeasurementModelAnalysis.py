############################################
# This file is used to compare the performance of the implementation of the measurement model
# in the Monte Carlo Localization algorithm. The measurement model is used to calculate the
# likelihood of a set of measurements given a pose of the robot.
# Model 0: Loops in both z and x
# Model 1: Loops in x
# Model 2: No loops
# The performance is compared by running each model 1000 times and measuring the time it takes
# to run each model. The results are printed to the console.
############################################

import yaml
import matplotlib.pyplot as plt
import numpy as np
from utils.grids import StochOccupancyGrid2D, DetOccupancyGrid2D
import os
import time

map_resolution = 0.05
map_height = 140
map_width = 140

def measurement_model0(z, x, theta_sens):
    """
    The measurement model for the LIDAR sensor. This is a likelihood model using distance to nearest neighbor
    See Probabilistic Robotics, Table 6.3 pg 172

    This model achieves the measurement model with a loop in z (measurement) and in x (particles)
    This implementation is slow...

    Args:
        z: The LIDAR measurement, a 1xN array where N is the number of measurements, we assume all measurements
            outside of the max range are already removed from this set
        x: The pose of the robot, a 3x1 array (x, y, theta)
        theta_sens: The angle of the sensor relative to the robot's frame, a 1xN array
    """
    q = 1
    for i in range(len(z)):
        x_meas = x[0] + z[i] * np.cos(x[2] + theta_sens[i])
        y_meas = x[1] + z[i] * np.sin(x[2] + theta_sens[i])

        x_grid = np.round(x_meas / map_resolution).astype(int)
        y_grid = np.round(y_meas / map_resolution).astype(int)

        if x_grid < 0 or x_grid >= map_height or y_grid < 0 or y_grid >= map_width:
            p = 1 / LIDAR_MAX_RANGE
        else:
            p = prob_lookup_table[x_grid, y_grid]
        q *= p
    return q

def measurement_model1( z, x, theta_sens):
    """
    The measurement model for the LIDAR sensor. This is a likelihood model using distance to nearest neighbor
    See Probabilistic Robotics, Table 6.3 pg 172

    This model achieves the measurement model with no loop in z. But, still requires only a single x input

    Args:
        z: The LIDAR measurement, a 1xN array where N is the number of measurements, we assume all measurements
            outside of the max range are already removed from this set
        x: The pose of the robot, a 3x1 array (x, y, theta)
        theta_sens: The angle of the sensor relative to the robot's frame, a 1xN array
    """
    x_meas = x[0] + z * np.cos(x[2] + theta_sens)
    y_meas = x[1] + z * np.sin(x[2] + theta_sens)

    # convert x_meas and y_meas to grid coordinates
    x_grid = np.round(x_meas / map_resolution).astype(int)
    y_grid = np.round(y_meas / map_resolution).astype(int)

    neg_x = np.where(x_grid < 0)
    out_of_range_x = np.where(x_grid >= map_height)
    neg_y = np.where(y_grid < 0)
    out_of_range_y = np.where(y_grid >= map_width)

    x_grid_norm = np.clip(x_grid, 0, map_height - 1)
    y_grid_norm = np.clip(y_grid, 0, map_width - 1)

    p = prob_lookup_table[x_grid_norm, y_grid_norm]

    p[neg_x] = 1 / LIDAR_MAX_RANGE
    p[out_of_range_x] = 1 / LIDAR_MAX_RANGE
    p[neg_y] = 1 / LIDAR_MAX_RANGE
    p[out_of_range_y] = 1 / LIDAR_MAX_RANGE

    return np.sum(np.power(p, 3))

def measurement_model2(z, x, theta_sens):
    """
    The measurement model for the LIDAR sensor. This is a likelihood model using distance to nearest neighbor
    See Probabilistic Robotics, Table 6.3 pg 172

    This model achieves the measurement model with no loop in z. But, still requires only a single x input

    Args:
        z: The LIDAR measurement, a 1xN array where N is the number of measurements, we assume all measurements
            outside of the max range are already removed from this set
        x: The pose of the robot, a 3xM array (x, y, theta), M is number of particles
        theta_sens: The angle of the sensor relative to the robot's frame, a 1xN array
    """
    n = len(z)  # number of measurements

    # Tile the x array to match the number of measurements
    x_tiled = np.tile(x[:, :, np.newaxis], (1, 1, n))

    # Calculate the x and y coordinates of the measurements in the map frame
    x_meas = x_tiled[0,:,:] + z * np.cos(x_tiled[2, :, :] + theta_sens)
    y_meas = x_tiled[1,:,:] + z * np.sin(x_tiled[2, :, :] + theta_sens)

    # convert x_meas and y_meas to grid coordinates
    x_grid = np.round(x_meas / map_resolution).astype(int)
    y_grid = np.round(y_meas / map_resolution).astype(int)

    # Get indices of out of range locations
    out_of_range_x = np.where((x_grid < 0) | (x_grid >= map_height))
    out_of_range_y = np.where((y_grid < 0) | (y_grid >= map_width))

    # Clip the grid coordinates to be within the map
    x_grid_norm = np.clip(x_grid, 0, map_height - 1)
    y_grid_norm = np.clip(y_grid, 0, map_width - 1)

    # Look up the probabilities from the precomputed table
    p = prob_lookup_table[x_grid_norm, y_grid_norm]

    # Set out of range locations to 1 / LIDAR_MAX_RANGE (these are unknown locations)
    p[out_of_range_x[0], out_of_range_x[1]] = 1 / LIDAR_MAX_RANGE
    p[out_of_range_y[0], out_of_range_y[1]] = 1 / LIDAR_MAX_RANGE

    # Instead of doing product of all probabilities, we sum p^3 as a heuristic
    return np.sum(np.power(p, 3), axis=1)

if __name__ == '__main__':
    LIDAR_MAX_RANGE = 16
    sigma_hit = 0.3
    z_hit = 0.75
    z_random = 0.25
    dist_lookup_table = np.load('lookup_table/mattbot_map.npy')
    unknown_indx = np.where(dist_lookup_table == -1)

    prob_lookup_table = z_hit / np.sqrt(2 * np.pi * (sigma_hit ** 2)) * np.exp(
        -0.5 * (dist_lookup_table) ** 2 / (sigma_hit ** 2)) + z_random / LIDAR_MAX_RANGE
    prob_lookup_table[unknown_indx] = 1 / LIDAR_MAX_RANGE

    np.random.seed(0)
    M = 500  # number of particles
    theta_sens = np.arange(-np.pi, np.pi, 0.1)
    z = np.random.rand(len(theta_sens)) * 16
    x = np.random.rand(3, M)
    x[0:2, :] = x[0:2, :] * 10
    x[2, :] = x[2, :] * 2 * np.pi - np.pi

    # Get current time
    t1 = time.time()
    for i in range(1000):
        measurement_model2(z, x, theta_sens)
    t2 = time.time()
    print("No Loop in M:")
    print((t2 - t1) / 1000)

    t1 = time.time()
    for i in range(1000):
        for m in range(M):
            measurement_model1(z, x[:, m], theta_sens)
    t2 = time.time()
    print("Loop in M:")
    print((t2 - t1)/ 1000)

    t1 = time.time()
    for i in range(50):
        for m in range(M):
            measurement_model0(z, x[:, m], theta_sens)
    t2 = time.time()
    print("Loop in M and z:")
    print((t2 - t1) / 50)
    a = 5