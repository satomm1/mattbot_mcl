############################################
# This file is used to analyze the performance of the motion model
# in the Monte Carlo Localization algorithm. The motion model is implemented requiring a loop in M
# and also in a way that requires no loop in M. The performance is compared by running each model 1000 times
# and measuring the time it takes to run each model. The results are printed to the console. We see approx 100x speed
# up when we remove the loop in M.
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

SQRT6DIV2 = np.sqrt(6)/2
alpha1=0.05
alpha2=0.05
alpha3=0.01
alpha4=0.001


def sample_motion_model(u, x_prev):
    """
    Samples the motion model based on odometry control. See Probabilistic Robotics, Table 5.6 pg 136

    Args:
        u: The control via odometry, a 3x2 array where the first column is the (x,y,theta) of previous time step from
            odometry and the second column is the (x,y,theta) of the current time step from odometry
        x_prev: The previous pose of the robot, a 3x1 array (x, y, theta)
    """
    delta_rot1 = np.arctan2(u[1, 1] - u[1, 0], u[0, 1] - u[0, 0]) - u[2, 0]
    delta_trans = np.sqrt((u[0, 1] - u[0, 0]) ** 2 + (u[1, 1] - u[1, 0]) ** 2)
    delta_rot2 = u[2, 1] - u[2, 0] - delta_rot1

    delta_rot1_hat = delta_rot1 - sample_normal(alpha1 * np.abs(delta_rot1) + alpha2 * delta_trans)
    delta_trans_hat = delta_trans - sample_normal(
        alpha3 * delta_trans + alpha4 * (np.abs(delta_rot1) + np.abs(delta_rot2)))
    delta_rot2_hat = delta_rot2 - sample_normal(alpha1 * np.abs(delta_rot2) + alpha2 * delta_trans)

    xp = x_prev[0] + delta_trans_hat * np.cos(x_prev[2] + delta_rot1_hat)
    yp = x_prev[1] + delta_trans_hat * np.sin(x_prev[2] + delta_rot1_hat)
    tp = x_prev[2] + delta_rot1_hat + delta_rot2_hat
    tp = wrap_theta(tp)

    return np.array([xp, yp, tp])

def sample_motion_model1(u, x_prev):
    """
    Samples the motion model based on odometry control. See Probabilistic Robotics, Table 5.6 pg 136

    Args:
        u: The control via odometry, a 3x2 array where the first column is the (x,y,theta) of previous time step from
            odometry and the second column is the (x,y,theta) of the current time step from odometry
        x_prev: The previous pose of the robot, a 3x1 array (x, y, theta)
    """
    M = x_prev.shape[1]

    delta_rot1 = np.arctan2(u[1, 1] - u[1, 0], u[0, 1] - u[0, 0]) - u[2, 0]
    delta_trans = np.sqrt((u[0, 1] - u[0, 0]) ** 2 + (u[1, 1] - u[1, 0]) ** 2)
    delta_rot2 = u[2, 1] - u[2, 0] - delta_rot1

    delta_rot1_hat = delta_rot1 - sample_normal(alpha1 * np.abs(delta_rot1) + alpha2 * delta_trans, m=M)
    delta_trans_hat = delta_trans - sample_normal(
        alpha3 * delta_trans + alpha4 * (np.abs(delta_rot1) + np.abs(delta_rot2)), m=M)
    delta_rot2_hat = delta_rot2 - sample_normal(alpha1 * np.abs(delta_rot2) + alpha2 * delta_trans, m=M)

    xp = x_prev[0,:] + delta_trans_hat * np.cos(x_prev[2,:] + delta_rot1_hat)
    yp = x_prev[1,:] + delta_trans_hat * np.sin(x_prev[2,:] + delta_rot1_hat)
    tp = x_prev[2,:] + delta_rot1_hat + delta_rot2_hat
    tp = wrap_theta(tp)

    return np.vstack((xp, yp, tp))

def sample_normal(b, m=None):
    """
    Samples a value from a normal distribution with mean 0 and standard deviation b
    """
    return np.random.normal(0, b, size=m)

def sample_triangular(b):
    """
    Samples a value from a triangular distribution with mean 0 and standard deviation b
    """
    return SQRT6DIV2 * (np.random.uniform(-b, b) + np.random.uniform(-b, b))

def wrap_theta(theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

if __name__ == "__main__":
    np.random.seed(0)
    M = 500  # number of particles
    x = np.random.rand(3, M)
    x[0:2, :] = x[0:2, :] * 10
    x[2, :] = x[2, :] * 2 * np.pi - np.pi

    u = np.array([[0, 0, 0], [.001, -0.001, 0.002]]).T

    t1 = time.time()
    for i in range(1000):
        sample_motion_model1(u, x)
    t2 = time.time()
    print("No Loop in M:")
    print((t2 - t1) / 1000)

    t1 = time.time()
    for i in range(1000):
        for m in range(M):
            sample_motion_model(u, x[:, m])
    t2 = time.time()
    print("Loop in M:")
    print((t2 - t1) / 1000)
