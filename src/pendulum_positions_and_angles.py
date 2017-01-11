# coding: utf8
"""
    determine center and angles of pendulum from CNN activation images
"""
# packages
from matplotlib import pyplot as plt
import numpy as np
import random as rd
from numpy.linalg import norm
from fit_circle import fit_circle, count_inliers

interseting_layers = [7, 124]
#interseting_layers = [6, 7, 8, 9, 13, 19, 28, 122, 124, 133, 152, 233]
#interseting_layers = [7]

def propagate_cluster(clusters, m, i, j, w, h, threshold):
    to_propagate = [(i,j)]
    to_return = [(i,j)]
    cardinal = 1
    cluster = clusters[i,j]
    i_max = i
    j_max = j
    while to_propagate!=[]:
        current = to_propagate.pop()
        clusters[current[0],current[1]] = cluster
        for k in range(max(0,current[0]-1), min(w,current[0]+2)):
            for l in range(max(0, current[1] - 1), min(h, current[1] + 2)):
                if (m[k,l] > threshold and clusters[k,l] == 0):
                    to_propagate.append((k,l))
                    to_return.append((k,l))
                    clusters[k,l] = cluster
                    cardinal += 1
                    if m[k,l] > m[i_max, j_max]:
                        i_max = k
                        j_max = l
    return to_return, cardinal, i_max, j_max

def find_biggest_clusters(m, threshold):
    w,h = m.shape
    clusters = np.zeros((w,h))
    num_clusters = 0
    biggest_cluster = []
    second_cluster = []
    card_second_cluster = 0
    card_biggest_cluster = 0
    highest_value = 0
    for i in range(w):
        for j in range(h):
            if (clusters[i,j]==0 and m[i,j]>threshold):
                num_clusters+=1
                clusters[i,j]=num_clusters
                list, card, i_max, j_max = propagate_cluster(clusters, m, i, j, w, h, threshold)
                if m[i_max,j_max] > highest_value:
                    card_second_cluster = card_biggest_cluster
                    card_biggest_cluster = card
                    second_cluster = biggest_cluster
                    biggest_cluster = list
                    highest_value = m[i_max,j_max]
    return biggest_cluster, card_biggest_cluster, second_cluster, card_second_cluster

def find_center_balls(m, threshold):
    cluster1, num_pixels1, cluster2, num_pixels2 = find_biggest_clusters(m, threshold)
    c1x = 0
    c1y = 0
    c2x = 0
    c2y = 0
    while cluster1 != []:
        (i, j) = cluster1.pop()
        c1x += i
        c1y += j
    surface_ratio = 1.0 / num_pixels1
    c1x *= surface_ratio
    c1y *= surface_ratio
    while cluster2 != []:
        (i, j) = cluster2.pop()
        c2x += i
        c2y += j
    surface_ratio = 1.0 / num_pixels2
    c2x *= surface_ratio
    c2y *= surface_ratio

    return np.array([c1x, c1y]), np.array([c2x,c2y])


def compute_radius(centers):
    n_centers = len(centers)
    threshold = 7
    step = 10

    radius = []

    for i in range(n_centers - step):
        c1 = centers[i]
        c2 = centers[i+step]

        if norm(c2 - c1) > threshold:
            diff = c2 - c1
            ortho = np.zeros(2)
            ortho[0] = diff[1]
            ortho[1] = -diff[0]
            ortho = ortho / norm(ortho)

            mid = 0.5 * (c1 + c2)

            radius.append([mid, ortho])

    return radius

def loss_function(radius, point):
    loss = 0
    for radius_params in radius:
        radius_point = radius_params[0]
        radius_vector = radius_params[1]

        x = point - radius_point

        loss += norm(x - x.dot(radius_vector) * radius_vector)**2

    return loss

# def grad_loss_function(radius, point):
#     g = (0,0)
#     for r in radius:
#         x = [point[i] - r[0][i] for i in range(2)]
#         xu = (x[0]*r[1][0]+x[1]*r[1][1])*r[1]
#         xv = [x[i] - xu[i] for i in range(2)]
#         step = (xv[0]*(1-point[0]*pow(r[1][0],2)), xv[1]*(1-point[1]*pow(r[1][1],2)))
#         g = [g[i] + step[i] for i in range(2)]
#     return g

def grad_exp(radius, point):
    epsilon = 0.0001

    point_x_eps = point + epsilon * np.array([1, 0])
    point_y_eps = point + epsilon * np.array([0, 1])

    grad = np.zeros(2)

    grad[0] = (loss_function(radius, point_x_eps) - loss_function(radius, point)) / epsilon
    grad[1] = (loss_function(radius, point_y_eps) - loss_function(radius, point)) / epsilon

    return grad

def find_attach_point(radius):
    attach_point = np.array([50, 50])

    # gradient algorithm parameters
    max_iter = 1000
    grad_stop = 0.001
    default_step = 100
    iter = 0

    grad_norm = 2 * grad_stop

    while (grad_norm > grad_stop and iter < max_iter):
        iter += 1
        step = default_step
        loss = loss_function(radius, attach_point)

        grad = grad_exp(radius, attach_point)

        while (loss_function(radius, attach_point - step * grad) > loss):
            step *= 0.5

        attach_point = attach_point - step * grad
        grad_norm = norm(grad)**2

    return attach_point

def determine_pendulum_positions_from_activations(activations_file=None, neuron_index=None):
    activations = np.load(activations_file)

    n_frames = activations.shape[0]
    centers = np.zeros((n_frames, 2))

    for f in range(n_frames):
        frame = activations[f, neuron_index]
        threshold = 0.5 * (np.max(frame) + np.min(frame))
        centers[f], _ = find_center_balls(frame, threshold)

    return centers

def ransac_circle(centers):
    n_frames = centers.shape[0]
    nb_iterations = 1000
    nb_points_to_fit = 6
    tolerance = 4

    max_inliers = 0
    best_inliers = []
    for iter in range(nb_iterations):
        indices = rd.shuffle(range(n_frames))[:nb_points_to_fit]
        rand_points = np.zeros((nb_points_to_fit,2))
        for p in range(nb_points_to_fit):
            pendulum_id = (rd.random()>0.5)
            rand_points[p,0] = centers[indices[p],pendulum_id*2]
            rand_points[p,1] = centers[indices[p],pendulum_id*2+1]
        center, radius = fit_circle(rand_points)
        inliers, nb_inliers = count_inliers(center, radius, centers, tolerance)
        if (nb_inliers > max_inliers):
            max_inliers = nb_inliers
            best_inliers = inliers
    center, radius = fit_circle(best_inliers)

    return center, radius

# def separate_balls(centers):
#     n_frames = centers.shape[0]
#     center, radius = ransac_circle(centers)
#     ball1 = np.zeros((n_frames, 2))
#     ball2 = np.zeros((n_frames, 2))
#     for f in range(n_frames):
#         to be continued

def determine_pendulum_angles_from_positions(positions):
    radius = compute_radius(positions)
    attach_point = find_attach_point(radius)
    angles = np.zeros(len(positions))
    angles[:] = np.arctan2(positions[:, 0] - attach_point[0], positions[:, 1] - attach_point[1])

    return angles

def determine_double_pendulum_angles_from_positions(positions):
    radius = compute_radius(positions)
    attach_point = find_attach_point(radius)
    angles = np.zeros(len(positions))
    angles[:] = np.arctan2(positions[:, 0] - attach_point[0], positions[:, 1] - attach_point[1])

    return angles
