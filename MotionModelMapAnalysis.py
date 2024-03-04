import numpy as np
import time

map_resolution = 0.05
map_height = 140
map_width = 140

resolution = 0.05
origin_x = 0
origin_y = 0
window_size = 5
width = 140
height = 140
thresh = 0.5

SQRT6DIV2 = np.sqrt(6)/2
alpha1=0.05
alpha2=0.05
alpha3=0.01
alpha4=0.001

probs = np.round(np.random.rand(height, width)) * 100

def sample_motion_model_with_map1(u, x_prev):
    """
    Samples the motion model with the map, does not require loop in M

    Args:
        x_prev: The previous pose of the robot, a 3xM array (x, y, theta)
    """
    pi = 1
    num_stuck = 0
    not_free_indx = np.arange(x_prev.shape[1])
    x = x_prev
    while np.sum(pi) < x_prev.shape[1]:
        if num_stuck > 10:
            x[:, not_free_indx] = x_prev[:, not_free_indx]
            return x

        x[:, not_free_indx] = sample_motion_model1(u, x_prev[:, not_free_indx])
        pi = is_free1(x)
        not_free_indx = np.where(pi == False)[0]

        num_stuck += 1
    return x

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

def wrap_theta(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def sample_normal(b, m=None):
    """
    Samples a value from a normal distribution with mean 0 and standard deviation b
    """
    return np.random.normal(0, b, size=m)

def is_free1(state):
    # combine the probabilities of each cell by assuming independence
    # of each estimation
    x, y = snap_to_grid1(state)
    grid_x = ((x - origin_x) / resolution).astype(int)
    grid_y = ((y - origin_y) / resolution).astype(int)

    # Now check probabilities
    half_size = int(round((window_size -1 ) /2))
    grid_x_lower = np.maximum(0, grid_x - half_size)
    grid_y_lower = np.maximum(0, grid_y - half_size)
    grid_x_upper = np.minimum(width, grid_x + half_size + 1)
    grid_y_upper = np.minimum(height, grid_y + half_size + 1)

    free_list = []
    for i in range(len(grid_x)):
        prob_window = probs[grid_y_lower[i]:grid_y_upper[i], grid_x_lower[i]:grid_x_upper[i]]
        p_total = np.prod(1. - np.maximum(prob_window / 100., 0.))
        free_list.append((1. - p_total) < thresh)

    return np.array(free_list)

def snap_to_grid1(x):
    return (resolution * np.round(x[0] / resolution), resolution * np.round(x[1] / resolution))

if __name__ == '__main__':
    M = 500  # number of particles
    x = np.random.rand(3, M)
    x[0:2, :] = x[0:2, :] * 10
    x[2, :] = x[2, :] * 2 * np.pi - np.pi

    u = np.array([[0, 0, 0], [.001, -0.001, 0.002]]).T

    x_new = sample_motion_model_with_map1(u, x)
    a = 1