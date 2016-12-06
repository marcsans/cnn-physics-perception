# coding: utf8

import theano
import numpy as np
import scipy as sp

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
            upscaled = sp.misc.imresize(fmap, size=(224, 224), mode="F", interp='bilinear')
            hypercolumns.append(upscaled)
    return np.asarray(hypercolumns)
