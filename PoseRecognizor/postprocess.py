import math
import os, re
import json
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import Counter



def angle_vec_plane(p1, p2, plane_normal = [0, 0, 1]):
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    # Compute the vector p1p2
    vec_p1p2 = p2 - p1
    
    # Compute the dot product of the vector p1p2 and the normal vector of the plane
    dot_product = np.dot(vec_p1p2, np.array(plane_normal))
    
    # Compute the magnitude of the vector p1p2
    magnitude_p1p2 = np.linalg.norm(vec_p1p2)
    
    # Compute the angle using the dot product and magnitude
    angle_rad = np.arccos(dot_product / magnitude_p1p2)
    angle_deg = np.degrees(angle_rad)
    
    return angle_rad

def angle_vecs(p1, p2, p3, p4):        
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    p4 = np.array(p4)

    if np.array_equal(p1,p3):
        p13_mid = (p1 + p2) / 2
        p1 = p13_mid
        p3 = p13_mid


    vec_p1p2 = p2 - p1
    vec_p3p4 = p4 - p3

    dot_product = np.dot(vec_p1p2, vec_p3p4)
    magnitude_p1p2 = np.linalg.norm(vec_p1p2)
    magnitude_p1p3 = np.linalg.norm(vec_p3p4)

    cos_theta = dot_product / (magnitude_p1p2 * magnitude_p1p3)
    angle_rad= np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    return angle_rad


def angle_dtw_cost(path, frames, template, angle_func=angle_vec_plane, angle_threshold=None, **kwargs):
    total_cost = 0.0
    costs = []
    for p in path:
        if angle_func.__name__ == "angle_vec_plane":
            pts_id = kwargs['one_vec']
            angle_diff = abs(angle_func(frames[p[0]][pts_id[0]], frames[p[0]][pts_id[1]], plane_normal=kwargs['plane_normal']) - angle_func(template[p[1]][pts_id[0]], template[p[1]][pts_id[1]], plane_normal=kwargs['plane_normal']))
        else: # angle_func.__name__ == "angle_vecs"
            # Angle difference between the same vector in input frame and template
            pts_id = kwargs['two_vecs']
            angle_diff = abs(angle_func(frames[p[0]][pts_id[0]], frames[p[0]][pts_id[1]], template[p[1]][pts_id[2]], template[p[1]][pts_id[3]]))

        if angle_threshold:
            cur_cost = angle_diff if angle_diff <= angle_threshold else 100
        else:
            cur_cost = angle_diff
        total_cost += cur_cost
        costs.append(cur_cost)
    return total_cost, costs

def extract_angle_feature(data):
    # Input: 3 key points coordinate which represent left arm or right arm, Shape: (6, 3)
    l1 = data[2, :] - data[1, :]
    l2 = data[0, :] - data[1, :]
    arm_angle = np.dot(l1 / np.linalg.norm(l1), l2 / np.linalg.norm(l2))

    return arm_angle

def jointMatching(path, pts_sequence, template_sequence, single_threshold):
    angle_cost_list = []
    speed_end_point = pts_sequence.shape[0] - 1
    speed_start_point = speed_end_point - 1 if path[0][0] == speed_end_point else path[0][0]
    template_sequence = np.array(template_sequence)
    # angle_speed = abs(extract_angle_feature(pts_sequence[speed_start_point]) - extract_angle_feature(pts_sequence[speed_end_point])) / (speed_end_point - speed_start_point) 
    angle_speed = (extract_angle_feature(pts_sequence[-1]) - extract_angle_feature(pts_sequence[5])) / (0.08 * 6)
    all_cost = 0
    # How many frame allow to over the single frame threshold
    frame_tolerance = 1 #TODO: check if 0 works
    for i in path:
        if (single_threshold):
            d = abs(extract_angle_feature(pts_sequence[i[0]]) - extract_angle_feature(template_sequence[i[1]]))
            #print("Angle matching distance: {}".format(d))
            if (d > single_threshold):
                if (frame_tolerance > 0):
                    frame_tolerance -= 1
                    cost = d
                else:
                    cost = 10
            else:
                cost = d
        angle_cost_list.append(cost)
        all_cost += cost

    return all_cost / len(path), angle_speed


def is_stable_movement(keypoints, threshold=0.2):
    keypoints = np.array(keypoints)
    avg_coord = np.mean(keypoints, axis=0) # (keypoints, 3)
    diff_coord = keypoints - avg_coord # (frames, keypoitns, 3)
    pt_dist_per_frame = np.linalg.norm(diff_coord, axis=2) # (frames, keypoints)
    avg_pt_dist = np.mean(pt_dist_per_frame, axis=0) # (keypoints,)
    is_stable = np.all(avg_pt_dist < threshold)

    return is_stable

def post_process(action_class, pts_sequence, label_dict, \
                 matched_path, matched_res, matched_label, matched_dis, \
                 match_angle=True, distance_threshold=None, reference_dis=None, **kwargs): # matched_dis = average frame distance
    matched_len = len(matched_path)
    matched_ids_cnt = Counter([pair[0] for pair in matched_path])
    is_same_frames = any(value >= len(matched_path)*0.8 for value in matched_ids_cnt.values())
    # Determine if the matched frames represent a stable movement
    is_stable = is_stable_movement(np.array(pts_sequence)[matched_path[0][0]:matched_path[-1][0]+1])

    # The reference distance only works on pressing now
    if not reference_dis:
        if action_class == 2 and reference_dis > distance_threshold:
            return matched_path, {"Msg": "No match", "Matched label": None, \
                                "Matched file": matched_res['file'], "Matched frame": matched_res['frame'], \
                                "Distance": reference_dis, "Avg distance": reference_dis / len(matched_path)}
    
    if matched_dis > distance_threshold:
        return matched_path, {"Msg": "No match", "Matched label": None, \
                            "Matched file": matched_res['file'], "Matched frame": matched_res['frame'], \
                            "Distance": matched_dis, "Avg distance": matched_dis / len(matched_path)}
    elif matched_path[0][0] == matched_path[-1][0] or matched_path[0][0] == matched_path[-2][0] or is_same_frames or len(matched_ids_cnt) <= 2:
        return matched_path, {"Msg": "One frame matched for too many times", "Matched label": label_dict[matched_label]}
    elif is_stable:
        return matched_path, {"Msg": "Stable movement detected", "Matched label": label_dict[matched_label]}

    else:
        if not match_angle:
            return matched_path, {"Msg": "Matched", "Matched label": label_dict[matched_label], \
                                "Matched file": matched_res['file'], "Matched frame": matched_res['frame'], \
                                "Distance": matched_dis * matched_len, "Avg distance": matched_dis}

        else:
            # Recognize swiping, zoom actions
            if action_class == 1:
                feature_func = angle_dtw_cost
                if matched_label == 3 or matched_label == 4 or matched_label == 5 or matched_label == 2: # Single arm (swipe)
                    if matched_label == 3 or matched_label == 4: # Left arm movement
                        kwargs['one_vec'] = [1,2]
                        kwargs['two_vecs'] = [0,2,0,2]

                    else: # matched_label == 2 or matched_label == 5: # Right arm movement
                        kwargs['one_vec'] = [4,5]
                        kwargs['two_vecs'] = [3,5,3,5]

                    angle_cost, costs = feature_func(matched_path, pts_sequence, matched_res['data'], angle_func=angle_vecs, angle_threshold=kwargs['angle_threshold'], two_vecs=kwargs['two_vecs'])

                    vecs_angle_cond = angle_cost <= matched_len * kwargs['avg_angle_threshold']
                    if vecs_angle_cond: 
                        return matched_path, {"Msg": "Matched", "Matched label": label_dict[matched_label], \
                                            "Matched file": matched_res['file'], "Matched frame": matched_res['frame'], \
                                            "Distance": matched_dis * matched_len, "Avg distance": matched_dis, \
                                            "Avg angle arm": angle_cost / matched_len}
                    else:
                        return matched_path, {"Msg": "Angle difference is over the threshold", "Matched label": label_dict[matched_label], \
                                            "Matched file": matched_res['file'], "Matched frame": matched_res['frame'], \
                                            "Distance": matched_dis * matched_len, "Avg distance": matched_dis, \
                                            "Avg angle arm": angle_cost / matched_len}
                    
                elif matched_label == 6 or matched_label == 7: # Two arms (zoom)
                    kwargs['one_vec_l'] = [1,2]
                    kwargs['one_vec_r'] = [4,5]
                    kwargs['two_vecs_l'] = [0,2,0,2]
                    kwargs['two_vecs_r'] = [3,5,3,5]
                    # Angle between Left shoulder and wrist, Right shoulder and wrist
                    angle_cost_l, costs_l = feature_func(matched_path, pts_sequence, matched_res['data'], angle_func=angle_vecs, angle_threshold=kwargs['angle_threshold'], two_vecs=kwargs['two_vecs_l'])
                    angle_cost_r, costs_r = feature_func(matched_path, pts_sequence, matched_res['data'], angle_func=angle_vecs, angle_threshold=kwargs['angle_threshold'], two_vecs=kwargs['two_vecs_r'])


                    vecs_angle_cond = angle_cost_l <= matched_len * kwargs['avg_angle_threshold'] and angle_cost_r <= matched_len * kwargs['avg_angle_threshold']
                    if vecs_angle_cond:
                        return matched_path, {"Msg": "Matched", "Matched label": label_dict[matched_label], \
                                            "Matched file": matched_res['file'], "Matched frame": matched_res['frame'], \
                                            "Distance": matched_dis * matched_len, "Avg distance": matched_dis, \
                                            "Avg angle left arm": angle_cost_l / matched_len, "Avg angle right arm": angle_cost_r / matched_len}
                    else:
                        return matched_path, {"Msg": "Angle difference is over the threshold", "Matched label": label_dict[matched_label], \
                                            "Matched file": matched_res['file'], "Matched frame": matched_res['frame'], \
                                            "Distance": matched_dis * matched_len, "Avg distance": matched_dis,\
                                            "Avg angle left arm": angle_cost_l / matched_len, "Avg angle right arm": angle_cost_r / matched_len}
            # Recognize Pressing action
            elif action_class == 2:
                feature_func = jointMatching
                if matched_label == 9: # Left hand pressing
                    angle_cost, hand_avg_velocity = feature_func(matched_path, np.array(pts_sequence)[:, :3], np.array(matched_res['data'])[:, :3], single_threshold=kwargs['angle_threshold'])
                else: # Right hand pressing
                    angle_cost, hand_avg_velocity = feature_func(matched_path, np.array(pts_sequence)[:, 3:], np.array(matched_res['data'])[:, 3:], single_threshold=kwargs['angle_threshold'])

                if hand_avg_velocity > -0.6:
                    return matched_path, {"Msg": "Angle velocity is over the threshold", "Matched label": {label_dict[matched_label]}, \
                                          "Matched file": matched_res['file'], "Matched frame": matched_res['frame'], \
                                          "Distance": matched_dis * matched_len, "Avg distance": matched_dis, \
                                          "Angle velocity": hand_avg_velocity}
                else:
                    if (angle_cost < kwargs['avg_angle_threshold']):
                        return matched_path, {"Msg": "Matched", "Matched label": label_dict[matched_label], \
                                              "Matched file": matched_res['file'], "Matched frame": matched_res['frame'], \
                                              "Distance": matched_dis * matched_len, "Avg distance": matched_dis}
                    else:
                        return matched_path, {"Msg": "Angle difference is over the threshold", "Matched label": {label_dict[matched_label]}, \
                                              "Matched file": matched_res['file'], "Matched frame": matched_res['frame'], \
                                              "Distance": matched_dis * matched_len, "Avg distance": matched_dis, \
                                               "Angle velocity": hand_avg_velocity}

