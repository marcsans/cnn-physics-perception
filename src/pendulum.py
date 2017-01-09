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

FPS = 31
dt = 1.0 / FPS

### SIMPLE PENDULUM ###
def simple_pendulum_derivatives(state, t, l, g):
    """
        state : [theta (angle), w (angular velocity) ]
        general equation
    """
    deriv = np.zeros(2)
    deriv[0] = state[1]
    deriv[1] = -(g / l) * sin(state[0])

    return deriv

def integrate_simple_pendulum(th_0, th_1, n_step, l=L1, g=G):
    w = (th_1 - th_0) / dt
    t = np.linspace(dt, n_step * dt, n_step)
    initial_state = np.array([th_0, w])
    y = integrate.odeint(simple_pendulum_derivatives, initial_state, t, args = (l, g))

    x1 = l * sin(y[:, 0])
    y1 = -l * cos(y[:, 0])

    return (y, x1, y1)

def perform_one_step_integration_for_simple_pendulum(th_0, th_1, l=L1, g=G, n_dt=1):
    DT = n_dt * dt
    return th_1 + DT * ((th_1 - th_0) / DT - DT * (g / l) * sin(th_0))

### DOUBLE PENDULUM
def double_pendulum_derivatives(state, t, m1, m2, l1, l2, g):
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
    MT = (m1 + m2)
    delta_theta = theta2 - theta1
    denominator1 = MT * l1 - m2 * l1 * cos(delta_theta) * cos(delta_theta)
    denominator2 = (l2 / l1) * denominator1

    # angular accelerations 1
    state_derivatives[1] = (m2 * l1 * w1 * w1 * sin(delta_theta) * cos(delta_theta) +
            m2 * g*sin(theta2) * cos(delta_theta) +
            m2 * l2 * w2 * w2 * sin(delta_theta) -
            MT * g*sin(theta1))
    state_derivatives[1] = state_derivatives[1] / denominator1

    # angular accelerations 2
    state_derivatives[3] = (-m2 * l2 * w2 * w2 * sin(delta_theta) * cos(delta_theta) +
            MT * g * sin(theta1) * cos(delta_theta) -
            MT * l1 * w1 * w1 * sin(delta_theta) -
            MT * g * sin(theta2))
    state_derivatives[3] = state_derivatives[3] / denominator2

    return state_derivatives

def integrate_double_pendulum(th1_0, th1_1, th2_0, th2_1, n_step=1):
    """
        given the two first angles for the two pendulum, integrates the doule pendulum equation
    """
    t = np.linspace(dt, n_step * dt, n_step)
    w1 = (th1_1 - th1_0) / dt
    w2 = (th2_1 - th2_0) / dt

    initial_state = [th1_1, w1, th2_1, w2]
    y = integrate.odeint(double_pendulum_derivatives, initial_state, t, args = (M1, M2, L1, L2, G))

    x1 = L1 * sin(y[:, 0])
    y1 = -L1 * cos(y[:, 0])

    x2 = L2 * sin(y[:, 2]) + x1
    y2 = -L2 * cos(y[:, 2]) + y1

    return (y, x1, y1, x2, y2)

### EXAMPLES ###
def run_simple_pendulum_example(l=L1):
    th_0 = -np.pi / 4
    th_1 = th_0
    n_step = 200

    (y, x1, y1) = integrate_simple_pendulum(th_0, th_1, n_step, l=l)

    animate_pendulum(y, x1, y1)


def run_double_pendulum_example_with_animation():
    t = np.arange(0.0, 20, dt)

    # initial angles
    th1_0 = np.pi / 2
    th1_1 = np.pi / 2
    th2_0 = -np.pi / 8
    th2_1 = -np.pi / 8
    n_step = 200

    (y, x1, y1, x2, y2) = integrate_double_pendulum(th1_0, th1_1, th2_0, th2_1, n_step=n_step)

    animate_pendulum(y, x1, y1, x2, y2)

# animation function

def animate_pendulum(y, x1, y1, x2=None, y2=None):
    is_simple_pendulum = (x2 is None or y2 is None)

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
        if (is_simple_pendulum):
            thisx = [0, x1[i]]
            thisy = [0, y1[i]]
        else:
            thisx = [0, x1[i], x2[i]]
            thisy = [0, y1[i], y2[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i * dt))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)), interval=25, blit=False, init_func=init)
    #ani.save('double_pendulum.mp4', fps=15)
    plt.show()
