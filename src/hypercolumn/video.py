# coding: utf8
from matplotlib import pyplot as plt
from matplotlib import animation

import cv2
import numpy as np

from keras.optimizers import SGD
from VGG_16 import VGG_16

from utils import extract_hypercolumn
from utils import get_activations

# Get neural network
model = VGG_16('vgg16_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

# Set up formatting for the movie files
FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title='Threshold test', artist='Steven Spielberg', comment='Rated 5 stars!')
writer = FFMpegWriter(fps=15, metadata=metadata)
fig = plt.figure()
ims=[]

# Capture the video
cap = cv2.VideoCapture('../../pendule.mp4')
f = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
    # redimensionnement de la frame pour qu' elle ait la bonne taille pour le réseau 224x 224
    frame = cv2.resize(frame, (224, 224))
    im = frame.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)

    # feed the image into VGG
    model.predict(im)

    # get features from the third layer
    feat = get_activations(model, 15, im)
    m = np.matrix((feat[0][0][7]>1000)*255.0, dtype=float)
    ims.append(m)
    f+=1
    print 'frame '+str(f)
    # if f>20:
    #     break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
with writer.saving(fig, "../../threshold.mp4", 100):
    for im in ims:
        plt.imshow(im)
        writer.grab_frame()
cv2.destroyAllWindows()

