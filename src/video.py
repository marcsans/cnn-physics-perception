# coding: utf8
"""
    reads a video and computes the activations for a given layer of the conv net VGG_16. Write the activations as a numpy array in a file
"""
# packages
from matplotlib import pyplot as plt
from matplotlib import animation
import cv2
import numpy as np
import subprocess as sp
from numpy.linalg import norm
from scipy.misc import imresize

#scripts
from VGG_16 import VGG_16
from utils import extract_hypercolumn
from utils import get_activations, get_video_number_of_frames



def write_activations_from_video(input_video='../pendule.mp4', input_video_shape=[852, 480], layer_index=15, activations_file='data/activations.npy'):
    image_size = input_video_shape[0] * input_video_shape[1] * 3

    convolutionnal_neural_network = VGG_16()
    n_frames = get_video_number_of_frames(input_video)
    activations = []

    cap = cv2.VideoCapture(input_video)

    for f in range(n_frames):
        ret, frame = cap.read()

        if frame is None:
            break

        image_CNN = cv2.resize(frame, (224, 224))
        image_CNN = image_CNN.transpose((2, 0, 1))
        image_CNN = np.expand_dims(image_CNN, axis=0)

        convolutionnal_neural_network.predict(image_CNN)

        activations.append(get_activations(convolutionnal_neural_network, layer_index, image_CNN)[0][0])

        print 'frame ', f, ' done'

    activations = np.array(activations)
    np.save(activations_file, activations)

def write_activations_video(activations_file='data/activations.npy'):
    activations = np.load(activations_file)
    n_frame = activations.shape[0]

    for layer in interseting_layers:
        print 'writing video for layer ', layer
        # Set up formatting for the movie files
        FFMpegWriter = animation.writers['ffmpeg']
        metadata = dict(title='Pendulum activation layer ' + str(layer), artist='Thiry, Sanselme')
        writer = FFMpegWriter(fps=15, metadata=metadata)

        output_video = '../activation_videos/' + str(layer) + 'threshold.mp4'
        fig = plt.figure()
        with writer.saving(fig, output_video, 100):
            for f in range(n_frame):
                print 'frame ', f
                im = activations[f, layer]
                plt.imshow(im)
                writer.grab_frame()

