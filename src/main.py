# coding: utf8
"""
    main script of the project
"""

import numpy as np
import matplotlib.pyplot as plt

import video
import pendulum
import pendulum_positions_and_angles
import learn_pendulum_parameter

#pendulum.run_simple_pendulum_example_with_animation(l=1.0)
#pendulum.run_simple_pendulum_example_with_animation(l=2.0)
#pendulum.run_double_pendulum_example_with_animation()

# video.write_activations_from_video(input_video='../pendule.mp4', layer_index=15, activations_file='data/activations_15.npy')
# video.write_activation_video(activations_file='data/activations_15.npy', neuron_index=7, output_video='../activation_videos/activation_video_layer_15_neuron_7.mp4', N_FPS=20, n_frames=40)

pendulum_positions = pendulum_positions_and_angles.determine_pendulum_positions_from_activations(activations_file='data/activations_15.npy', neuron_index=7)
#pendulum_positions = pendulum_positions_and_angles.determine_pendulum_positions_from_activations(activations_file='data/activations_1.npy', neuron_index=27)
pendulum_angles = pendulum_positions_and_angles.determine_pendulum_angles_from_positions(pendulum_positions)
plt.plot(pendulum_angles)
plt.title('pendulum angles')
plt.show()


(l, k) = learn_pendulum_parameter.learn_length_from_sequence_v1(pendulum_angles - 0.5 * np.pi)
print 'pendulum length ', l, ' . friction coeff ', k
