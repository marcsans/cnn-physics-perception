from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg


def double_pendulum_derivatives(state, t):
    """
        the state variable contains the angles and angular veolcities [theta1, w1, theta2, w2]
        function returns the derivative of the state which is angular velocities and accelerations [w1, a1, w2, a2]
    """

    state_derivatives = np.zeros_like(state)

    # trivial derivatives
    state_derivatives[0] = state[1]
    state_derivatives[2] = state[3]

    # simplify notations
    theta1 = state[0]
    w1 = state[1]
    theta2 = state[2]
    w2 = state[3]
    MT = (M1 + M2)
    delta_theta = theta2 - theta1
    denominator1 = MT * L1 - M2 * L1 * cos(delta_theta) * cos(delta_theta)
    denominator2 = (L2 / L1) * denominator1

    # angular accelerations 1
    state_derivatives[1] = (M2 * L1 * w1 * w1 * sin(delta_theta) * cos(delta_theta) +
            M2 * G*sin(theta2) * cos(delta_theta) +
            M2 * L2 * w2 * w2 * sin(delta_theta) -
            MT * G*sin(theta1))
    state_derivatives[1] = state_derivatives[1] / denominator1

    # angular accelerations 2
    state_derivatives[3] = (-M2 * L2 * w2 * w2 * sin(delta_theta) * cos(delta_theta) +
            MT * G * sin(theta1) * cos(delta_theta) -
            MT * L1 * w1 * w1 * sin(delta_theta) -
            MT * G * sin(theta2))
    state_derivatives[3] = state_derivatives[3] / denominator2

    return state_derivatives

def integrate_double_pendulum(th1_0, th1_1, th2_0, th2_1, n_fps=25, n_step=1):
    dt = 1.0 / n_fps
    t = np.linspace(dt, n_step * dt, n_step)
    w1 = (th1_1 - th1_0) / dt
    w2 = (th2_1 - th2_0) / dt

    initial_state = np.radians([th1_1, w1, th2_1, w2])
    y = integrate.odeint(double_pendulum_derivatives, initial_state, t)

    x1 = L1 * sin(y[:, 0])
    y1 = -L1 * cos(y[:, 0])

    x2 = L2 * sin(y[:, 2]) + x1
    y2 = -L2 * cos(y[:, 2]) + y1

    return (y, x1, y1, x2, y2)

# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.05
t = np.arange(0.0, 20, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 120.0
w1 = 0.0
th2 = -10.0
w2 = 0.0

# initial state
state = np.radians([th1, w1, th2, w2])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(double_pendulum_derivatives, state, t)

# compute the coordinates
x1 = L1 * sin(y[:, 0])
y1 = -L1 * cos(y[:, 0])

x2 = L2 * sin(y[:, 2]) + x1
y2 = -L2 * cos(y[:, 2]) + y1

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i * dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
                              interval=25, blit=False, init_func=init)

#ani.save('double_pendulum.mp4', fps=15)
plt.show()
