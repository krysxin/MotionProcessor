import sys, os, re
import json, yaml
import numpy as np
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt

from oneEuroFilter import OneEuroFilter
from preprocess import normalizePoints

# Load the configuration from the YAML file
def load_yaml(config_file):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except yaml.YAMLError as error:
        print(f"Error parsing YAML file: {error}")


def load_json(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data

def on_mqtt_connect(client, userdata, flags, rc):
    if rc==0:
        print("Connection successful. Returned code:",rc)
    else:
        print("Bad connection. Returned code:",rc)

def get_next_filename(filename, folder):
    filenames = os.listdir(folder)
    existing_ids = []
    base_filename, ext = os.path.splitext(filename)

    for filename in filenames:
        match = re.search(rf"{base_filename}_(\d{{2}})\.\w+$", filename)
        if match:
            existing_id = int(match.group(1))
            existing_ids.append(existing_id)

    if existing_ids:
        new_id = max(existing_ids) + 1
    else:
        new_id = 1

    new_filename = f"{base_filename}_{new_id:02d}{ext}"
    return new_filename


def write_data2json(data, savepath, indent=4):
    folder = os.path.dirname(savepath)
    filename = os.path.basename(savepath)
    
    # Check if the file already exists
    if os.path.exists(savepath):
        # Extract the file extension
        # base, ext = os.path.splitext(filename)
        
        new_filename = get_next_filename(filename, folder)        
        savepath = os.path.join(folder, new_filename)
    
    # Save the file
    with open(savepath, 'w') as json_file:
        json.dump(data, json_file, indent=indent, separators=(", ", ": "))

def has_nested_list(lst):
    return any(isinstance(element, list) for element in lst)

def transform1D(frame, use_noise_filter = False, noise_filter = None, t=None):
    coords = []
    if not has_nested_list(frame):
        coords = frame
    else:
        for pt in frame[:-4]:
            for i in pt[0:3]:
                coords.append(i)

    # Only used for local test and plotting purposes
    if use_noise_filter: 
        coords = noise_filter(t, coords)

    return coords

def plot_coords(filter_x, filter_x_hat):
    x = np.array(filter_x)
    x_hat = np.array(filter_x_hat)
    t = np.linspace(0, 50, 50)
    fig, ax = plt.subplots()
    ax.set(
        xlabel="$t$",
        ylabel="$x$",
    )
    signal, = ax.plot(t, x[50:100, 0], 'r-')
    filtered, = ax.plot(t, x_hat[50:100, 0], '-')
    plt.show()

def templates_mirror_flip(keypoints, origin=[0,0,0], normal=[0, -1, 0], rotate=True):
    # input [14, 3]

    keypoints = np.array(keypoints)
    body_idx = [2, 5, 8, 11]
    center = np.sum(keypoints[i] for i in body_idx) / 4

    # Get the normal direction where the body facing 
    v1 = keypoints[11] - keypoints[2]
    v2 = keypoints[8] - keypoints[5]

    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)
    # The transformation flips points across the plane defined by the normal and Z-axis
    mirror_matrix = np.eye(3) - 2 * np.outer(normal, normal)

    # Apply the transformation to all keypoints
    mirrored_keypoints = center + np.dot(keypoints - center, mirror_matrix)

    for i in [[2, 5], [3, 6], [4, 7], [8, 11], [9, 12], [10, 13]]:
        temp = np.copy(mirrored_keypoints[i[0]])
        mirrored_keypoints[i[0]] = mirrored_keypoints[i[1]]
        mirrored_keypoints[i[1]] = temp
    mirrored_keypoints = normalizePoints(mirrored_keypoints, origin=origin, vec=normal, rotate=rotate)
    return  mirrored_keypoints

