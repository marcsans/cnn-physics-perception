import pendulum
import numpy as np
import learn_pendulum_parameter

print 'test simple pendulum'
pendulum.run_simple_pendulum_example(l=1.0)
pendulum.run_simple_pendulum_example(l=2.0)

print 'test double pendulum animation'
#pendulum.run_double_pendulum_example_with_animation()

print 'test parameter learning'
l = 1.1
th_0 = np.pi / 4
th_1 = th_0
n_step = 50

(y, x1, y1) = pendulum.integrate_simple_pendulum(th_0, th_1, n_step, l=l)
angle_sequence = y[:, 0]

l_learn = learn_pendulum_parameter.learn_length_from_pendulum_angle_sequence(angle_sequence)
print 'theoritical value ', l
print 'learnt value value ', l_learn
