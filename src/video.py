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



def write_activations_from_video(input_video='../pendule.mp4', layer_index=15, activations_file='data/activations.npy'):

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

    print 'activation file ' + activations_file + ' written'

def write_activation_video(activations_file='data/activations.npy', neuron_index=0, output_video='../activation_videos/activation_video.mp4', N_FPS=15, n_frames=None):
    activations = np.load(activations_file)

    if (n_frames is None):
        n_frames = activations.shape[0]
    else:
        n_frames = min(n_frames, activations.shape[0])


    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=N_FPS)

    fig = plt.figure()

    with writer.saving(fig, output_video, 100):
        for f in range(n_frames):
            print 'writing frame ', f
            im = activations[f, neuron_index]
            plt.imshow(im)
            writer.grab_frame()

