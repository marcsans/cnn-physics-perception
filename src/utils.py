# coding: utf8
"""
    TODO
"""

import theano
import numpy as np
import scipy as scp
import subprocess

from keras import backend as K

def get_activations(model, layer_idx, X_batch):
    get_activations_func = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output,])
    activations = get_activations_func([X_batch,0])
    return activations

# extract hypercolumn
def extract_hypercolumn(model, layer_indexes, instance):
    layers = [model.layers[li].output for li in layer_indexes]
    get_feature = theano.function([model.layers[0].input], layers,allow_input_downcast=False)
    feature_maps = get_feature(instance)
    hypercolumns = []
    for convmap in feature_maps:
        for fmap in convmap[0]:
            upscaled = scp.misc.imresize(fmap, size=(224, 224), mode="F", interp='bilinear')
            hypercolumns.append(upscaled)
    return np.asarray(hypercolumns)

def get_video_number_of_frames(video):
    get_n_frame_command = [
        'ffprobe',
        '-v', 'error', # This hides "info" output (version info, etc) which makes parsing easier
        '-count_frames', # Count the number of frames per stream and report it in the corresponding stream section
        '-select_streams', 'v:0', # Select only the video stream
        '-show_entries', 'stream=nb_read_frames', # Show only the number of read frames
        '-of', 'default=nokey=1:noprint_wrappers=1', # Set output format (aka the "writer") to default, do not print the key of each field (nokey=1), and do not print the section header and footer (noprint_wrappers=1)
        video
    ]

    get_n_frame_pipe = subprocess.Popen(get_n_frame_command, stdout=subprocess.PIPE)
    n_frames = min(int(get_n_frame_pipe.stdout.readline()), 30)

    get_n_frame_pipe.stdout.close()
    get_n_frame_pipe.wait()
    del get_n_frame_pipe

    return n_frames
