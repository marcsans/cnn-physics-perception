import pendulum
import numpy as np
import learn_pendulum_parameter
import matplotlib.pyplot as plt

print 'test simple pendulum'
#pendulum.run_simple_pendulum_example(l=1.0)
#pendulum.run_simple_pendulum_example(l=2.0)

print 'test double pendulum animation'
# pendulum.run_double_pendulum_example_with_animation()

print 'test parameter learning'
l = 1.1
th_0 = np.pi / 4
th_1 = th_0
th_2 = pendulum.perform_one_step_integration_for_simple_pendulum(th_0, th_1, l=l)

learn_pendulum_parameter.plot_F(th_0, th_1, th_2, l)

l_learn = learn_pendulum_parameter.learn_length_from_three_angle(th_0, th_1, th_2, l_init=1.0)
print 'theoritical value ', l
print 'learnt value value ', l_learn

angles = np.loadtxt('data/angle_sequence.txt')
plt.plot(angles)
plt.show()



(l_pend, length) = learn_pendulum_parameter.learn_length_from_sequence(angles)
print 'pendulum length', l_pend
