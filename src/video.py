# coding: utf8
# packages
from matplotlib import pyplot as plt
from matplotlib import animation
import cv2
import numpy as np
import subprocess as sp
from keras.optimizers import SGD
from numpy.linalg import norm
from scipy.misc import imresize

#scripts
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
    surface_ratio = 1.0 / num_pixels
    cx *= surface_ratio
    cy *= surface_ratio
    return cx, cy

def compute_radius(centers):
    n_centers = len(centers)
    threshold = 7
    step = 10

    radius = []

    for i in range(n_centers - step):
        c1 = centers[i]
        c2 = centers[i+step]

        if norm(c2 - c1) > threshold:
            diff = c2 - c1
            ortho = np.zeros(2)
            ortho[0] = diff[1]
            ortho[1] = -diff[0]
            ortho = ortho / norm(ortho)

            mid = 0.5 * (c1 + c2)

            radius.append([mid, ortho])

    return radius

def loss_function(radius, point):
    loss = 0
    for radius_params in radius:
        radius_point = radius_params[0]
        radius_vector = radius_params[1]

        x = point - radius_point

        loss += norm(x - x.dot(radius_vector) * radius_vector)**2

    return loss

# def grad_loss_function(radius, point):
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

    point_x_eps = point + epsilon * np.array([1, 0])
    point_y_eps = point + epsilon * np.array([0, 1])

    grad = np.zeros(2)

    grad[0] = (loss_function(radius, point_x_eps) - loss_function(radius, point)) / epsilon
    grad[1] = (loss_function(radius, point_y_eps) - loss_function(radius, point)) / epsilon

    return grad

def find_attach_point(radius):
    attach_point = np.array([50, 50])

    # gradient algorithm parameters
    max_iter = 1000
    grad_stop = 0.001
    default_step = 100
    iter = 0

    grad_norm = 2 * grad_stop

    while (grad_norm > grad_stop and iter < max_iter):
        iter += 1
        step = default_step
        loss = loss_function(radius, attach_point)

        grad = grad_exp(radius, attach_point)

        while (loss_function(radius, attach_point - step * grad) > loss):
            step *= 0.5

        attach_point = attach_point - step * grad
        grad_norm = norm(grad)**2

    return attach_point

def write_activation_video_2(input_video='../pendule.mp4', input_video_shape=[852, 480], output_video='../threshold_2.mp4'):
    image_size = input_video_shape[0] * input_video_shape[1] * 3

    # Get neural network
    model = VGG_16('data/vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    # Set up formatting for the movie files
    FFMPEG_BIN = 'ffmpeg'
    read_video_command = [
        FFMPEG_BIN,
        '-i', input_video, # input video
        '-f', 'image2pipe', #use a pipe
        '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', # raw RGB output
        '-' # the pipe is usd by another programm
    ]
    write_video_command = [
        FFMPEG_BIN,
        '-y', # (optional) overwrite output file if it exists
        '-f', 'rawvideo', '-vcodec','rawvideo', # raw video format
        '-s', '56x56', # size of one frame
        '-pix_fmt', 'rgb24', # RGB
        '-r', '24', # frames per second
        '-i', '-', # The imput comes from a pipe
        '-an', # Tells FFMPEG not to expect any audio
        '-vcodec', 'mpeg4',
        output_video
    ]

    # open the read and write pipes
    read_video_pipe = sp.Popen(read_video_command, stdout=sp.PIPE, bufsize=10**9)
    write_video_pipe = sp.Popen(write_video_command, stdin=sp.PIPE)

    frame_number = 0
    continue_condition = True
    while continue_condition:
        try:
            frame_number += 1

            raw_image = read_video_pipe.stdout.read(image_size)
            image =  np.fromstring(raw_image, dtype='uint8')
            image = image.reshape((input_video_shape[1], input_video_shape[0], 3))

            # image to feed into the CNN
            im_CNN = imresize(image, (224, 224, 3))
            im_CNN = im_CNN.transpose((2, 0, 1))
            im_CNN = np.expand_dims(im_CNN, axis=0)

            # feed the image into VGG
            model.predict(im_CNN)

            # get features from the third layer
            threshold = 1000
            feat = get_activations(model, 15, im_CNN)
            print 'feat shape ', len(feat), ' ', len(feat[0]), ' ', len(feat[0][0]), ' ', len(feat[0][0][0])

            activations = np.matrix((feat[0][0][7]), dtype=float)
            activation_thresh = 255 * (activations > threshold)

            activation_image = np.zeros(activations.shape + (3, ))
            activation_image[:, :, 0] = activation_thresh
            activation_image[:, :, 1] = activation_thresh
            activation_image[:, :, 2] = activation_thresh

            activation_image = activation_image.astype('uint8')
            write_video_pipe.stdin.write(activation_image.tostring())

            print 'frame ', frame_number, ' done'
        except Exception:
            print 'stop'
            continue_condition = False


    read_video_pipe.stdout.flush()
    read_video_pipe.stdout.close()
    read_video_pipe.wait()

    del read_video_pipe

    write_video_pipe.stdin.close()
    write_video_pipe.wait()

    del write_video_pipe

def write_activation_video(adress_video_input='../pendule.mp4', adress_video_output='../threshold.mp4'):
    # Get neural network
    model = VGG_16('data/vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    # Set up formatting for the movie files
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Pendulum activation', artist='Thiry, Sanselme', comment='Activation of a particular neurone thresholded')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    ims=[]

    fig = plt.figure()
    # Capture the video
    cap = cv2.VideoCapture(adress_video_input)
    f = 0
    centers = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        # redimensionnement de la frame pour qu' elle ait la bonne taille pour le réseau 224x 224
        shape = frame.shape
        frame = cv2.resize(frame, (224, 224))
        print 'frame shape ', frame.shape
        im = frame.transpose((2, 0, 1))
        im = np.expand_dims(im, axis=0)
        print 'im shape ', im.shape

        # feed the image into VGG
        model.predict(im)

        # get features from the third layer
        threshold = 1000
        feat = get_activations(model, 15, im)
        m = np.matrix((feat[0][0][7]), dtype=float)
        centers.append(find_center_ball(m, threshold))
        ims.append((m>threshold)*255.0)
        f+=1
        print 'frame '+str(f)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

    with writer.saving(fig, adress_video_output, 100):
        for im in ims:
            plt.imshow(im)
            writer.grab_frame()

    cv2.destroyAllWindows()

    centers = np.array(centers)
    np.savetxt('data/centers.txt', centers)

def determine_pendulum_centers_from_activation_video(activation_video_url='../../threshold.mp4'):
    """
        NOT WORKING YET
    """
    cap = cv2.VideoCapture(activation_video_url)
    f = 0
    centers = []

    while (cap.isOpened()):
        ret, frame = cap.read()
        f += 1
        print 'frame ' + str(f) + '...'
        if frame is None:
            break
        # sum values of RGB channels to get a 2D array
        frame = np.sum(frame, axis=2)

        threshold = 0.5 * (np.max(frame) + np.min(frame))

        centers.append(find_center_ball(frame, threshold))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print 'done'
    cap.release()

    cv2.destroyAllWindows()

    centers = np.array(centers)
    np.savetxt('data/centers2.txt', centers)

#write_activation_video()
write_activation_video_2()
#determine_pendulum_centers_from_activation_video()
centers = np.loadtxt('data/centers.txt')
centers[:, 1] *= 852.0 / 480
radius = compute_radius(centers)
attach_point = find_attach_point(radius)

plt.figure(1)
plt.scatter(centers[:, 1], -centers[:, 0], color='blue')
plt.scatter(attach_point[1], -attach_point[0], color='red')
plt.title('pendulum positions and attach point')
plt.show()

angles = np.zeros(len(centers))
angles[:] = np.arctan2(centers[:, 0] - attach_point[0], centers[:, 1] - attach_point[1])

plt.figure(1)
plt.plot(angles)
plt.title('pendulum angles')
plt.show()

np.savetxt('data/angle_sequence.txt', angles)
