# coding: utf8
from matplotlib import pyplot as plt
from matplotlib import animation

import cv2
import numpy as np

from keras.optimizers import SGD
from VGG_16 import VGG_16

from utils import extract_hypercolumn
from utils import get_activations

def propagate_cluster(clusters, m, i, j, w, h, threshold):
    to_propagate = [(i,j)]
    to_return = [(i,j)]
    cardinal = 1
    cluster = clusters[i,j]
    i_max = i
    j_max = j
    while to_propagate!=[]:
        current = to_propagate.pop()
        clusters[current[0],current[1]] = cluster
        for k in range(max(0,current[0]-1), min(w,current[0]+2)):
            for l in range(max(0, current[1] - 1), min(h, current[1] + 2)):
                if (m[k,l] > threshold and clusters[k,l] == 0):
                    to_propagate.append((k,l))
                    to_return.append((k,l))
                    clusters[k,l] = cluster
                    cardinal += 1
                    if m[k,l] > m[i_max, j_max]:
                        i_max = k
                        j_max = l
    return to_return, cardinal, i_max, j_max

def find_biggest_clusters(m, threshold):
    w,h = m.shape
    clusters = np.zeros((w,h))
    num_clusters = 0
    biggest_cluster = []
    card_biggest_cluster = 0
    highest_value = 0
    for i in range(w):
        for j in range(h):
            if (clusters[i,j]==0 and m[i,j]>threshold):
                num_clusters+=1
                clusters[i,j]=num_clusters
                list, card, i_max, j_max = propagate_cluster(clusters, m, i, j, w, h, threshold)
                if m[i_max,j_max] > highest_value:
                    card_biggest_cluster = card
                    biggest_cluster = list
                    highest_value = m[i_max,j_max]
    return biggest_cluster, card_biggest_cluster

def find_center_ball(m, threshold):
    cluster, num_pixels = find_biggest_clusters(m, threshold)
    cx = 0
    cy = 0
    while cluster != []:
        (i,j) = cluster.pop()
        cx += i
        cy += j
    cx /= num_pixels
    cy /= num_pixels
    return cx, cy


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
    threshold = 1000
    feat = get_activations(model, 15, im)
    m = np.matrix((feat[0][0][7]), dtype=float)
    print find_center_ball(m, threshold)
    ims.append((m>threshold)*255.0)
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