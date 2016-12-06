# coding: utf8
from matplotlib import pyplot as plt

import theano
import cv2
import numpy as np
import scipy as sp

from keras import backend as K
from keras.optimizers import SGD
from VGG_16 import VGG_16

def get_activations(model, layer_idx, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output,])
    activations = get_activations([X_batch,0])
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

model = VGG_16('vgg16_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

# lecture et redimensionnement d'une image pour qu' elle ait la bonne taille pour le réseau 224x 224
im_original = cv2.resize(cv2.imread('madruga.jpg'), (224, 224))
im = im_original.transpose((2,0,1))
im = np.expand_dims(im, axis=0)
im_converted = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)
plt.imshow(im_converted)

# feed the image into VGG
out = model.predict(im)
plt.plot(out.ravel())

# get features from the third layer
feat = get_activations(model, 3, im)
plt.title('features from the third layer')
plt.imshow(feat[0][0][2])
plt.show()

# get features from the 15th layer
plt.title('features from the 15th layer')
feat = get_activations(model, 15, im)
plt.imshow(feat[0][0][13])
plt.show()

# layers to extract
layers_extract = [3, 8]
hc = extract_hypercolumn(model, layers_extract, im)

# average hc values
ave = np.average(hc.transpose(1, 2, 0), axis=2)
plt.imshow(ave)
plt.title('average hypercolumn extracted from layer 3 and 8')
plt.show()

#idem
layers_extract = [22, 29]
hc = extract_hypercolumn(model, layers_extract, im)
ave = np.average(hc.transpose(1, 2, 0), axis=2)
plt.imshow(ave)
plt.title('average hypercolumn extracted from layer 22 and 29')
plt.show()
