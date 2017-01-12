import numpy as np

def dist(x,y):
    return np.sqrt(pow(x[0]-y[0],2)+pow(x[1]-y[1],2))

def norm(x):
    return np.sqrt(pow(x[0],2)+pow(x[1],2)+pow(x[2],2))

def cost_to_minimize(center, radius, points):
    nb_points = points.shape[0]
    cost = 0
    for p in range(nb_points):
        cost += pow(dist(center, points[p])-radius,2)

    return cost

def grad_cost(center, radius, points):
    eps = 0.001
    cost_x = cost_to_minimize(center+np.array([eps,0.]),radius,points)
    cost_y = cost_to_minimize(center+np.array([0.,eps]),radius,points)
    cost_rad = cost_to_minimize(center,radius+eps,points)
    cost = cost_to_minimize(center,radius,points)
    grad = np.array([(cost_x-cost)/eps, (cost_y - cost) / eps, (cost_rad - cost)/eps])

    return grad

def fit_circle(centers):
    nb_points = centers.shape[0]
    center_estimate =  np.sum(centers,0)/nb_points
    radius_estimate = dist(centers[0],center_estimate)
    nb_iterations = 10000
    grad_lim = 0.0001

    iter = 0
    grad = np.array([1.,1.,1.])
    while (iter < nb_iterations and norm(grad)>grad_lim):
        iter +=1
        grad = grad_cost(center_estimate,radius_estimate,centers)
        step = -10 * grad
        while cost_to_minimize(center_estimate,radius_estimate,centers) > cost_to_minimize(center_estimate+step[:2],radius_estimate+step[2],centers):
            step /= 2
        center_estimate += step[:2]
        radius_estimate += step[2]

    return center_estimate, radius_estimate

def count_inliers(center, radius, points, tolerance):
    nb_points = points.shape[0]
    nb_inliers = 0
    inliers=[]
    for p in range(nb_points):
        if (np.abs(dist(center,points[p][:2])-radius)<tolerance):
            nb_inliers+=1
            inliers.append(points[p][:2])
    for p in range(nb_points):
        if (np.abs(dist(center,points[p][3:])-radius)<tolerance):
            nb_inliers+=1
            inliers.append(points[p][2:])

    return inliers, nb_inliers