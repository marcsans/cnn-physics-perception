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

def compute_total_cost(thetas,l,n_dt=1):
    theta_l = pendulum.perform_n_steps_integration_for_simple_pendulum(thetas[0], thetas[1], nb_steps=(len(thetas)-2), l=l, n_dt=1)
    th_expected = (thetas[2:]+thetas[1:-1])/2
    cost = (theta_l[1:] - th_expected)**2

    return np.sum(cost)/(len(thetas) - 2)

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

def learn_length_from_n_angles(thetas, l_init=1.0, step=1):
    """
        gradient descent with backtrack line search to minimise F(l)
    """
    # gradient descent parameters
    n_iteration_max = 1000
    diff = 10**(-2)
    epsilon = 10**(-6) # step to compute numerical gradient

    # initialize length
    l = l_init
    iter = 0

    # compute arrival points
    th_expected = (thetas[2:]+thetas[1:-1])/2

    continue_condition = True
    while continue_condition:
        delta = 1000 # gradient descent step
        iter += 1

        # compute numerical derivative
        l_eps = l + epsilon
        theta_l_eps = pendulum.perform_n_steps_integration_for_simple_pendulum(thetas[0], thetas[1], l=l_eps, nb_steps=(len(thetas)-2), n_dt=step)
        theta_l = pendulum.perform_n_steps_integration_for_simple_pendulum(thetas[0], thetas[1], l=l, nb_steps=(len(thetas)-2), n_dt=step)
        d_theta_l = (theta_l_eps[1:] - theta_l[1:]) / epsilon

        # compute gradient
        gradient = 2 * np.sum(d_theta_l * (theta_l[1:] - th_expected))/(len(thetas) - 2)

        # update l
        Fl = compute_total_cost(thetas, l)

        while (compute_total_cost(thetas, l - delta * gradient) > Fl or (l - delta * gradient<0)):
            delta = 0.5 * delta

        l = l - delta * gradient
        continue_condition = (iter < n_iteration_max and abs(gradient) > diff)

    plt.figure(2)
    plt.plot(theta_l[1:])

    return l

def learn_length_from_sequence_v1(angle_sequence, step=1):
    g = 9.81
    dt = 1.0 / 31

    angle_velocities_mid = 1.0 / dt * (angle_sequence[1:] - angle_sequence[0:-1])
    angle_sequence_mid = 0.5 * (angle_sequence[1:] + angle_sequence[0:-1])

    l = g * dt * np.sum(np.sin(angle_sequence_mid[:-1])**2) / np.sum(np.sin(angle_sequence_mid[:-1] * (angle_velocities_mid[:-1] - angle_velocities_mid[1:])))

    return l

def learn_length_from_sequence(angle_sequence, step=1):
    length = 1.
    lengths = []
    step_sequence = 5
    N = len(angle_sequence)

    for i  in range(3, N, step_sequence):
        length = learn_length_from_n_angles(angle_sequence[:i], length, step=step)
        lengths.append(length)
        print str(i * 100 / N) + "%"

    print length
    print lengths
    plt.figure(3)
    plt.plot(range(len(lengths)), lengths)
    plt.show()

    return (length, lengths)

