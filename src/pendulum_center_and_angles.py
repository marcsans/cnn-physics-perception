# coding: utf8
"""
    TODO
"""
# packages
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import norm

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
    card_biggest_cluster = 0
    highest_value = 0
    for i in range(w):
        for j in range(h):
            if (clusters[i,j]==0 and m[i,j]>threshold):
                num_clusters+=1
                clusters[i,j]=num_clusters
                list, card, i_max, j_max = propagate_cluster(clusters, m, i, j, w, h, threshold)
                if m[i_max,j_max] > highest_value:
                    card_biggest_cluster = card
                    biggest_cluster = list
                    highest_value = m[i_max,j_max]
    return biggest_cluster, card_biggest_cluster

def find_center_ball(m, threshold):
    cluster, num_pixels = find_biggest_clusters(m, threshold)
    cx = 0
    cy = 0
    while cluster != []:
        (i,j) = cluster.pop()
        cx += i
        cy += j
    surface_ratio = 1.0 / num_pixels
    cx *= surface_ratio
    cy *= surface_ratio

    return np.array([cx, cy])

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

def determine_pendulum_centers():
    activations = np.load('data/activations.npy')

    n_frames = activations.shape[0]

    for layer in interseting_layers:
        centers = np.zeros((n_frames, 2))

        for f in range(n_frames):
            frame = activations[f, layer]
            threshold = 0.5 * (np.max(frame) + np.min(frame))

            centers[f] = find_center_ball(frame, threshold)

        np.savetxt('data/centers_layer_' + str(layer) + '.txt', centers)

def plot_pendulum_positions_and_angles():
    for layer in interseting_layers:
        centers = np.loadtxt('data/centers_layer_' + str(layer) + '.txt')
        radius = compute_radius(centers)
        attach_point = find_attach_point(radius)

        plt.figure(str(layer) + ' positions')
        plt.axis('equal')
        plt.scatter(centers[:, 1], -centers[:, 0], color='blue')
        plt.scatter(attach_point[1], -attach_point[0], color='red')
        plt.title('pendulum positions')
        plt.figure(str(layer) + ' angles')
        angles = np.zeros(len(centers))
        angles[:] = np.arctan2(centers[:, 0] - attach_point[0], centers[:, 1] - attach_point[1])
        plt.plot(angles)
        plt.title('pendulum angles')

    plt.show()

#determine_pendulum_centers_from_activation_video()
determine_pendulum_centers()
plot_pendulum_positions_and_angles()
