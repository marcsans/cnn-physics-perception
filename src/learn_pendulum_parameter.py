# coding: utf8
"""
    function used to learn the pendulum parameters from the angle sequence
"""
from numpy import sin, cos, abs
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import pendulum

def compute_F(th_0, th_1, th_2, l, n_dt=1):
    theta_l = pendulum.perform_one_step_integration_for_simple_pendulum(th_0, th_1, l=l, n_dt=1)
    return (theta_l - th_2)**2

def plot_F(th_0, th_1, th_2, l):
    n_points = 1000
    l_values = np.linspace(l-1, l+1 , n_points)

    F = np.zeros(n_points)

    for i, l_value in enumerate(l_values):
        F[i] = compute_F(th_0, th_1, th_2, l_value)

    plt.figure(1)
    plt.plot(l_values, F)
    plt.show()

def learn_length_from_three_angle(th_0, th_1, th_2, l_init=1.0, step=1):
    """
        gradient descent with backtrack line search to minimise F(l)
    """
    # gradient descent parameters
    n_iteration_max = 10000
    diff = 10**(-4)
    epsilon = 10**(-6) # step to compute numerical gradient

    # initialize length
    l = l_init
    iter = 0

    continue_condition = True
    while continue_condition:
        delta = 1000 # gradient descent step
        iter += 1

        # compute numerical derivative
        l_eps = l + epsilon
        theta_l_eps = pendulum.perform_one_step_integration_for_simple_pendulum(th_0, th_1, l=l_eps, n_dt=step)
        theta_l = pendulum.perform_one_step_integration_for_simple_pendulum(th_0, th_1, l=l, n_dt=step)
        d_theta_l = (theta_l_eps - theta_l) / epsilon

        # compute gradient
        gradient = 2 * d_theta_l * (theta_l - th_2)

        # update l
        Fl = compute_F(th_0, th_1, th_2, l)

        while (compute_F(th_0, th_1, th_2, l - delta * gradient) > Fl):
            delta = 0.5 * delta

        l = l - delta * gradient
        continue_condition = (iter < n_iteration_max and abs(gradient) > diff)

    return l

def learn_length_from_sequence(angle_sequence, step=1):
    lengths = []
    l = 0
    N = len(angle_sequence) - 2

    average_diff = 0
    for i  in range(0, N - step):
        average_diff += abs(angle_sequence[i + step] - angle_sequence[i])

    average_diff *= 1.0 / (N - step)

    for i  in range(0, N - 2 * step):
        th_0 = angle_sequence[i]
        th_1 = angle_sequence[i + step]
        th_2 = angle_sequence[i + 2 * step]

        monotony_condition = ((th_2 > th_1 and th_1 > th_0) or (th_2 < th_1 and th_1 < th_0))
        diff_condition = (abs(th_2 - th_1) > average_diff and abs(th_1 - th_0) > average_diff)

        if (monotony_condition and diff_condition):
            length = learn_length_from_three_angle(th_0, th_1, th_2, step=step)
            lengths.append(length)

    plt.figure(1)
    plt.hist(lengths, bins=50)
    plt.show()

    return (sum(lengths) / N, lengths)
