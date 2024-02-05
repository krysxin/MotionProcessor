import math
import os, re
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial.distance import euclidean, cdist

from preprocess import normalizePoints, cart2polar
from postprocess import post_process
from utils import load_json, get_next_filename, write_data2json

from fastdtw import fastdtw




def dtw1(pts_sequence, template_sequence, axes=None, single_frame_threshold=2.0, **kwargs):
    pts_sequence_arr = np.array(pts_sequence).reshape(len(pts_sequence), -1)
    template_sequence_arr = np.array(template_sequence).reshape(len(template_sequence), -1)
    if axes:
    # Extract selected axes coordinates only
        num_coords = pts_sequence_arr.shape[-1]
        pts_sequence_arr = np.concatenate([pts_sequence_arr[:, i:num_coords:3] for i in axes], axis=1)
        template_sequence_arr = np.concatenate([template_sequence_arr[:, i:num_coords:3] for i in axes], axis=1)

    penalty = 100
    if single_frame_threshold:
        def distance_func(x, y): return penalty if np.linalg.norm(x-y) > single_frame_threshold else np.linalg.norm(x-y)
    else:
        def distance_func(x, y): return np.linalg.norm(x-y)
    distance, path = fastdtw(pts_sequence_arr, template_sequence_arr, dist=distance_func)
    
    path_length = [distance_func(pts_sequence_arr[i[0]], template_sequence_arr[i[1]]) for i in path]
    
    return distance, path, path_length


#------DTW3----------------------------------------------------------

# ---Helper functions for DTW3---------------------------------------
def angle_difference(angle1, angle2):
    # Normalize the angles to be in the range [0, 2π]
    angle1 = angle1 % (2 * math.pi)
    angle2 = angle2 % (2 * math.pi)
    
    # Calculate the absolute angular difference
    abs_diff = np.abs(angle1 - angle2)
    
    # Take the minimum of the absolute difference and 2π - absolute difference
    angle_diff = np.minimum(abs_diff, 2 * math.pi - abs_diff)
    # angle_diff = abs_diff
    
    return angle_diff

def polar_dist(x,y,threshold=2.0, use_r = False):
    """
    x, y: array of polar coordinates, shape: (1, (keypoints_num)*3)
    """
    delta_theta = np.abs(x[1::3] - y[1::3])
    delta_phi = angle_difference(x[2::3], y[2::3])
    delta_r = np.abs(x[0::3] - y[0::3]) if use_r else np.zeros_like(delta_theta)

    delta_total = np.concatenate((delta_r, delta_theta, delta_phi))

    if threshold:
        return 100 if np.linalg.norm(delta_total) > threshold else np.linalg.norm(delta_total)
    else:
        return np.linalg.norm(delta_total)

def cart_dist(x, y, threshold=2.0):
    # Compute the distance between points x and y under cartesian coordinates
    penalty = 100
    if threshold:
        return penalty if np.linalg.norm(x-y) > threshold else np.linalg.norm(x-y)
    else:
        return np.linalg.norm(x-y)

def get_min(m0, m1, m2, i, j):
    if m0 < m1:
        if m0 < m2:
            return i - 1, j, m0
        else:
            return i - 1, j - 1, m2
    else:
        if m1 < m2:
            return i, j - 1, m1
        else:
            return i - 1, j - 1, m2
# ---End of Helper functions for DTW3---------------------------------------

def dtw3(x, y, axes=None, single_frame_threshold=2.0, use_polar=False, open_begin=True, open_end=False): #partial dtw
    x = np.array(x).reshape(len(x), -1) # (n, 6, 3) => (n, 18)
    y = np.array(y).reshape(len(y), -1)

    if axes:
        # Extract selected axes coordinates only
        num_coords = x.shape[-1]
        x = np.concatenate([x[:, i:num_coords:3] for i in axes], axis=1)
        y = np.concatenate([y[:, i:num_coords:3] for i in axes], axis=1)


    Tx = len(x)
    Ty = len(y)

    C = np.zeros((Tx, Ty))
    B = np.zeros((Tx, Ty, 2), int)

    # Set the distance function as 'polar_dist' if using polar coords
    dist_func = polar_dist if use_polar else cart_dist


    C[0, 0] = dist_func(x[0], y[0], threshold=single_frame_threshold)
    path_length = []

    for i in range(Tx):
        #--------For open-begin------------
        if open_begin:
            C[i, 0] = dist_func(x[i], y[0], threshold=single_frame_threshold)
            B[i, 0] = [i, 0]
        #--------------------------------
        else:
            C[i, 0] = C[i - 1, 0] + dist_func(x[i], y[0], threshold=single_frame_threshold)
            B[i, 0] = [i-1, 0]

    for j in range(1, Ty):
        C[0, j] = C[0, j - 1] + dist_func(x[0], y[j], threshold=single_frame_threshold)
        B[0, j] = [0, j - 1]

    for i in range(1, Tx):
        for j in range(1, Ty):
            pi, pj, m = get_min(C[i - 1, j],
                                C[i, j - 1],
                                C[i - 1, j - 1],
                                i, j)
            C[i, j] = dist_func(x[i], y[j], threshold=single_frame_threshold) + m
            B[i, j] = [pi, pj]
    #--------For open-end------------
    if open_end:
        t_end = np.argmin(C[:,-1])
        cost = C[t_end, -1]
        
        path = [[t_end, Ty - 1]]
        i = t_end
        j = Ty - 1
    #--------------------------------
    else:
        cost = C[-1, -1]
        path = [[Tx-1, Ty - 1]]
        i, j = Tx-1, Ty-1

        # Retrieve shortest matching path
    prev = [Tx-1, Ty-1]
    while True:
        cur = B[i, j]
        if np.array_equal(cur, prev) or np.array_equal(cur, np.array([-1, 0])): break
        path.append(cur)
        path_length.append(C[i, j] - C[B[i, j][0], B[i, j][1]])
        i, j = cur.astype(int)
        prev = cur

        
    path = np.flip(path, axis=0).tolist()
    path_length.append(C[i, j])
    path_length = np.flip(path_length).tolist()
    avg_cost = cost / len(path)
        
    return cost, path, path_length #, avg_cost

#--------------------END of DTW3-------------------------------------   

def top_n_references(matched_templates, n=10):
    # sort top n matching path by sum of the matching distance
    top_n_templates = sorted(matched_templates, key=lambda x: x[2])[:n]
    top_n_avg_dis = sum(m[2] for m in top_n_templates) / n
    top_n_avg_len = sum(len(m[1]) for m in top_n_templates) / n
    # print(top_n_avg_dis)
    # print(top_n_templates)
    # top_n references distance = sum(top_n matching distance) / avg matching length
    return top_n_avg_dis / top_n_avg_len

def recognizor_dynamic(model, action_class, pts_sequence, templates, label_dict, \
                        single_frame_threshold=None, distance_threshold=None, \
                        open_begin=True, open_end=False, \
                        match_angle=False, use_polar=False, axes=None, n_references=None, \
                        post_process_func=post_process, **kwargs):
    """
    Arguments:
        single_frame_threshold: threshold for a single frame, used for dtw distance
        distance_threshold: threshold for frame average distance, used to restrict the total distance of the matcing path
        match_angle: boolean variable to decide whether to add angle check after the best match is found,
                    if it is set to True, kwargs will have two arguments angle_threshold and avg_angle_threshold

    Return:
        Matching path, and matching result
    """

    matched_label = -1
    min_distance = math.inf
    distance = []
    matched_res = None
    p = []
    matched_p = None

    for entry in templates:
        d, path, path_length = model(pts_sequence, entry['data'], axes=axes, single_frame_threshold=single_frame_threshold, use_polar=use_polar, open_begin=open_begin, open_end=open_end)
        avg_d = d / len(path)
        p.append(path)
        distance.append((entry['file'], path_length, d))
        if avg_d < min_distance:
            min_distance = avg_d
            matched_label = entry['label']
            matched_res = entry
            matched_p = path

    if (n_references > 0):
        reference_dis = top_n_references(distance, n=n_references)
    else: 
        reference_dis = None
    dtw_path, recognition_result = post_process_func(action_class, pts_sequence, label_dict, \
                                                    matched_p, matched_res, matched_label, min_distance, \
                                                    match_angle=match_angle, distance_threshold=distance_threshold, reference_dis=reference_dis, **kwargs)
    
    return dtw_path, recognition_result

    
    
    
        

    
   
