from numpy import sin, cos, abs
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import pendulum

FPS = 25
dt = 1.0 / FPS

def learn_length_from_pendulum_angle_sequence(angle_sequence):
    th_0 = angle_sequence[0]
    th_1 = angle_sequence[1]

    n_iteration_max = 10000

    # gradient algo parameters
    diff = 10**(-4)
    epsilon = 10**(-6) # step to compute numerical gradient
    delta = 0.001 # gradient descent step

    # initialize length
    l = 1.0
    iter = 0

    continue_condition = True
    while continue_condition:
        iter += 1

        l_eps = l + epsilon

        (states_l, x1, x2) = pendulum.integrate_simple_pendulum(th_0, th_1, 3, l=l)
        (states_l_eps, x1_eps, x2_eps) = pendulum.integrate_simple_pendulum(th_0, th_1, 3, l=l_eps)

        theta = angle_sequence[2]
        theta_l = states_l[2, 0]
        theta_l_eps = states_l_eps[2, 0]
        G_theta_l = (theta_l_eps - theta_l) / epsilon

        gradient = 2 * G_theta_l * (theta_l - theta)

        # update l
        l = l - delta * gradient
        print l
        continue_condition = (iter < n_iteration_max and abs(gradient) > diff)

    return l
