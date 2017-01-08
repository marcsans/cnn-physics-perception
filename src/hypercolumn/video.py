# coding: utf8
from matplotlib import pyplot as plt
from matplotlib import animation
import cv2
import numpy as np
from keras.optimizers import SGD
from VGG_16 import VGG_16
from utils import extract_hypercolumn
from utils import get_activations

# parameters
write_video = False
adress_video_input = '../../pendule.mp4'
adress_video_output = "../../threshold.mp4"

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

def dist(c1,c2):
    return np.sqrt(pow(c2[0]-c1[0],2)+pow(c2[1]-c1[1],2))

def norm_square(v):
    return pow(v[0],2)+pow(v[1],2)

def find_radius(centers):
    l = len(centers)
    radius = []
    threshold = 7
    step = 10
    for i in range(l-step):
        c1 = centers[i]
        c2 = centers[i+step]
        if dist(c1,c2) > threshold:
            diff = [c2[i] - c1[i] for i in range(2)]
            ortho = ((diff)[1],-(diff)[0])
            ortho = ortho / np.sqrt(norm_square(ortho))
            mid= [(c1[i] + c2[i])/2 for i in range(2)]
            radius.append((mid,ortho))
    return radius

def loss(radius, point):
    l = 0
    for r in radius:
        x = [point[i] - r[0][i] for i in range(2)]
        xu = (x[0]*r[1][0]+x[1]*r[1][1])*r[1]
        xv = [x[i] - xu[i] for i in range(2)]
        l += norm_square(xv)
    return l

# def grad_loss(radius, point):
#     g = (0,0)
#     for r in radius:
#         x = [point[i] - r[0][i] for i in range(2)]
#         xu = (x[0]*r[1][0]+x[1]*r[1][1])*r[1]
#         xv = [x[i] - xu[i] for i in range(2)]
#         step = (xv[0]*(1-point[0]*pow(r[1][0],2)), xv[1]*(1-point[1]*pow(r[1][1],2)))
#         g = [g[i] + step[i] for i in range(2)]
#     return g

def grad_exp(radius, point):
    epsilon = 0.0001
    point_delta1 = [point[i] + (1-i)*epsilon for i in range(2)]
    point_delta2 = [point[i] + i*epsilon for i in range(2)]
    return ((loss(radius, point_delta1)-loss(radius,point))/epsilon, (loss(radius, point_delta2)-loss(radius,point))/epsilon)

def find_attach_point(radius):
    attach_point = (50,50)
    max_iter =1000
    grad_stop = 0.001
    grad = (1,1)
    iter = 0
    while (norm_square(grad) > grad_stop and iter < max_iter):
        iter += 1
        #grad = grad_loss(radius, attach_point)
        grad = grad_exp(radius,attach_point)
        l = loss(radius, attach_point)
        step = [-10*grad[i] for i in range(2)]
        backline_iter = 0
        while (l < loss(radius,  [attach_point[i] + step[i] for i in range(2)])):
            step = [step[i]/2 for i in range(2)]
            backline_iter+=1
            if backline_iter > 10000:
                print "error with backline search "
        attach_point = [attach_point[i] + step[i] for i in range(2)]
    return attach_point


# # Get neural network
# model = VGG_16('vgg16_weights.h5')
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='categorical_crossentropy')
#
# # Set up formatting for the movie files
# if write_video:
#     FFMpegWriter = animation.writers['ffmpeg']
#     metadata = dict(title='Pendulum activation', artist='Thiry, Sanselme', comment='Activation of a particular neurone thresholded')
#     writer = FFMpegWriter(fps=15, metadata=metadata)
#     ims=[]
#
# fig = plt.figure()
# # Capture the video
# cap = cv2.VideoCapture(adress_video_input)
# f = 0
# centers = []
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     if frame is None:
#         break
#     # redimensionnement de la frame pour qu' elle ait la bonne taille pour le réseau 224x 224
#     frame = cv2.resize(frame, (224, 224))
#     im = frame.transpose((2, 0, 1))
#     im = np.expand_dims(im, axis=0)
#
#     # feed the image into VGG
#     model.predict(im)
#
#     # get features from the third layer
#     threshold = 1000
#     feat = get_activations(model, 15, im)
#     m = np.matrix((feat[0][0][7]), dtype=float)
#     centers.append(find_center_ball(m, threshold))
#     if write_video:
#         ims.append((m>threshold)*255.0)
#     f+=1
#     print 'frame '+str(f)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
#
# if write_video:
#     with writer.saving(fig, adress_video_output, 100):
#         for im in ims:
#             plt.imshow(im)
#             writer.grab_frame()
#
# cv2.destroyAllWindows()

centers = [(24, 11), (24, 11), (24, 11), (24, 11), (24, 11), (24, 11), (24, 11), (24, 11), (24, 11), (24, 11), (24, 11), (25, 11), (26, 12), (28, 12), (30, 13), (32, 14), (34, 16), (36, 18), (38, 20), (39, 22), (40, 25), (41, 28), (41, 30), (40, 32), (39, 35), (37, 36), (36, 38), (34, 39), (32, 40), (31, 41), (30, 42), (29, 42), (28, 43), (28, 43), (28, 43), (28, 42), (29, 42), (31, 41), (32, 41), (34, 40), (35, 38), (37, 37), (39, 35), (40, 33), (41, 31), (41, 29), (41, 26), (40, 24), (39, 22), (38, 19), (36, 18), (34, 16), (33, 15), (31, 14), (30, 13), (29, 12), (28, 12), (28, 12), (28, 12), (28, 12), (29, 13), (30, 13), (32, 14), (33, 15), (35, 17), (36, 18), (38, 20), (39, 22), (40, 24), (41, 26), (41, 29), (40, 31), (40, 33), (39, 35), (38, 36), (36, 37), (35, 39), (34, 40), (33, 40), (32, 41), (31, 41), (31, 41), (31, 41), (32, 41), (32, 41), (33, 40), (34, 39), (35, 38), (37, 37), (38, 36), (39, 34), (40, 32), (40, 30), (41, 29), (41, 26), (40, 24), (39, 22), (38, 21), (37, 19), (36, 18), (34, 16), (33, 15), (32, 15), (32, 14), (31, 14), (31, 14), (31, 14), (31, 14), (32, 15), (33, 15), (34, 16), (35, 17), (37, 19), (38, 20), (39, 22), (40, 24), (40, 25), (40, 27), (40, 29), (40, 31), (39, 33), (39, 35), (38, 36), (37, 37), (35, 38), (35, 39), (34, 40), (33, 40), (33, 41), (33, 41), (33, 41), (33, 40), (34, 40), (35, 39), (36, 38), (37, 37), (38, 36), (39, 35), (40, 33), (40, 31), (40, 29), (40, 27), (40, 25), (40, 24), (39, 22), (38, 20), (37, 19), (36, 18), (35, 17), (34, 16), (34, 16), (33, 15), (33, 15), (33, 15), (33, 15), (34, 16), (34, 16), (35, 17), (36, 18), (37, 19), (38, 20), (39, 22), (40, 23), (40, 25), (41, 27), (41, 29), (40, 30), (40, 32), (39, 33), (39, 35), (38, 36), (37, 37), (36, 38), (35, 39), (35, 39), (34, 40), (34, 40), (34, 40), (34, 39), (35, 39), (35, 38), (36, 38), (37, 37), (38, 36), (39, 34), (39, 32), (40, 31), (40, 29), (40, 28), (40, 26), (40, 24), (40, 23), (39, 21), (38, 20), (37, 19), (36, 18), (35, 18), (35, 17), (34, 17), (34, 17), (34, 17), (34, 17), (35, 17), (35, 17), (36, 18), (37, 19), (37, 20), (38, 21), (39, 22), (40, 23), (40, 25), (40, 26), (40, 28), (41, 30), (40, 31), (40, 33), (39, 34), (38, 35), (38, 36), (37, 37), (36, 38), (36, 38), (35, 39), (35, 39), (35, 39), (35, 39), (36, 38), (37, 38), (37, 37), (38, 36), (39, 35), (39, 33), (40, 32), (40, 31), (40, 29), (41, 28), (41, 26), (40, 25), (40, 23), (39, 22), (39, 21), (38, 20), (37, 19), (36, 18), (36, 18), (36, 18), (35, 17), (35, 17), (35, 17), (36, 18), (36, 18), (37, 19), (37, 19), (38, 20), (39, 21), (39, 23), (40, 24), (40, 25), (40, 27), (40, 28), (40, 30), (40, 31), (40, 32), (39, 34), (38, 35), (38, 36), (37, 37), (37, 37), (36, 38), (36, 38), (36, 38), (36, 38), (36, 38), (37, 37), (37, 37), (38, 36), (38, 35), (39, 34), (39, 33), (40, 32), (40, 31), (40, 29), (40, 28), (40, 26), (40, 25), (40, 23), (39, 22), (39, 21), (38, 20), (37, 19), (37, 19), (37, 19), (36, 18), (36, 18), (36, 18), (36, 18), (36, 18), (37, 19), (37, 19), (38, 20), (38, 21), (39, 22), (39, 23), (40, 24), (40, 26), (40, 27), (40, 28), (40, 30), (40, 31), (40, 32), (39, 33), (39, 34), (38, 35), (38, 36), (37, 37), (37, 37), (37, 37), (37, 37), (37, 37), (37, 37), (37, 37), (38, 36), (38, 35), (39, 34), (39, 33), (40, 32), (40, 31), (40, 30), (40, 29), (40, 27), (40, 26), (40, 25), (39, 23), (39, 22), (38, 21), (38, 21), (37, 20), (37, 19), (36, 19), (36, 19), (36, 19), (36, 19), (36, 19), (36, 19), (37, 19), (37, 20), (38, 21), (38, 22), (39, 23), (39, 23), (40, 25), (40, 26), (40, 28), (40, 29), (40, 30), (40, 31), (40, 32), (39, 33), (38, 34), (38, 35), (38, 36), (37, 36), (37, 36), (37, 37), (37, 36), (37, 37), (37, 36), (37, 36), (38, 35), (38, 35), (39, 34), (39, 33), (40, 31), (40, 31), (40, 29), (40, 28), (40, 27), (40, 26), (40, 24), (39, 23), (39, 22), (38, 21), (38, 21), (38, 20), (37, 20), (37, 19), (37, 19), (37, 19), (37, 19), (37, 19), (37, 20), (38, 20), (38, 21), (39, 21), (39, 22), (39, 23), (39, 24), (40, 25), (40, 27), (40, 28), (40, 29), (40, 30), (40, 31), (40, 32), (39, 33), (39, 34), (38, 35), (38, 35), (38, 36), (38, 36), (38, 36), (38, 36), (38, 36), (38, 35), (38, 35), (38, 34), (39, 34), (39, 33), (39, 32), (40, 31), (40, 30), (40, 29), (40, 28), (40, 26), (40, 25), (40, 24), (39, 23), (39, 22), (39, 21), (38, 21), (38, 20), (37, 20), (37, 20), (37, 19), (37, 19), (37, 20), (37, 20), (38, 20), (38, 21), (39, 21), (39, 22), (39, 23), (40, 24), (40, 25), (40, 26), (40, 27), (40, 29), (40, 30), (40, 31), (40, 31), (39, 32), (39, 33), (39, 34), (38, 35), (38, 35), (38, 35), (38, 35), (38, 35), (38, 35), (38, 35), (38, 35), (38, 34), (39, 34), (39, 33), (40, 32)]
radius = find_radius(centers)
point = find_attach_point(radius)
print point