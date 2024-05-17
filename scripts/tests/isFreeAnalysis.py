import numpy as np
import time

resolution = 0.05
origin_x = 0
origin_y = 0
window_size = 5
width = 140
height = 140
thresh = 0.5

probs = np.round(np.random.rand(height, width)) * 100

def is_free(state):
    # combine the probabilities of each cell by assuming independence
    # of each estimation
    x, y = snap_to_grid(state)
    grid_x = int((x - origin_x) / resolution)
    grid_y = int((y - origin_y) / resolution)

    # Now check probabilities
    half_size = int(round((window_size -1 ) /2))
    grid_x_lower = max(0, grid_x - half_size)
    grid_y_lower = max(0, grid_y - half_size)
    grid_x_upper = min(width, grid_x + half_size + 1)
    grid_y_upper = min(height, grid_y + half_size + 1)

    prob_window = probs[grid_y_lower:grid_y_upper, grid_x_lower:grid_x_upper]
    p_total = np.prod(1. - np.maximum(prob_window / 100., 0.))

    return (1. - p_total) < thresh

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

def snap_to_grid(x):
    return (resolution*round(x[0]/resolution), resolution*round(x[1]/resolution))

def snap_to_grid1(x):
    return (resolution * np.round(x[0] / resolution), resolution * np.round(x[1] / resolution))

if __name__ == '__main__':
    np.random.seed(0)
    M = 500  # number of particles
    x = np.random.rand(3, M)
    x[0:2, :] = x[0:2, :] * 10
    x[2, :] = x[2, :] * 2 * np.pi - np.pi

    print(is_free1(x[0:2, :]))


