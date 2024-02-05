import math, random
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# ref https://github.com/sluyters/QuantumLeap/blob/master/backend/src/implementation/recognizers/dynamic/p3dollar/p3dollar/p3dollar.js
# Implementation of the preprocess of the point cloud, and the cloud matching algorithm


def GreedySequenceMatch(sq1, sq2):
    # Sequence matching - multiframe
    d = 0
    n = len(sq1)
    for i in range(n):
        d += GreedyCloudMatch(sq1[i], sq2[i])
    d_avg = d / n
    return d_avg

def SequenceDistance(sq1, sq2):
    # Convert sequences to NumPy arrays
    seq1_arr = np.array(sq1)
    seq2_arr = np.array(sq2)

    # Calculate the distance between the two sequences
    distance = np.linalg.norm(seq1_arr - seq2_arr)

    return distance

def GreedyCloudMatch(pts1, pts2):
    # Point matching - single frame
    e = 0.50
    step = math.floor(math.pow(len(pts1), 1.0 - e))
    min_d = math.inf
    for i in range(0, len(pts1), step):
        d1 = CloudDistance(pts1, pts2, i)
        d2 = CloudDistance(pts2, pts1, i)
        min_d = min(min_d, min(d1, d2))
    return min_d



def CloudDistance(pts1, pts2, start):
    matched = [False for _ in range(len(pts1))]
    sum = 0
    i = start
    while True:
        index = -1
        min_d = math.inf
        for j in range(len(matched)):
            if not matched[j]:
                d = Distance(pts1[i], pts2[j])
                if (d < min_d):
                    min_d = d
                    index = j
        matched[index] = True
        weight = 1 - ((i - start + len(pts1)) % len(pts1)) / len(pts1)
        sum += weight * min_d
        i = (i+1) % len(pts1)
        if i == start:
            break
    return sum




def Resample(points, n):
    # Preporcess - normalization of the points
    I = PathLength(points) / (n-1)
    D = 0.0
    num_newpoints = 0
    newpoints = [points[0]]
    for i in range(1, len(points)):
        if len(newpoints) <= n:
            d = Distance(points[i-1], points[i])
            if D+d >= I:
                print(f'Insert point between id {i-1} and {i}')
                qx = points[i-1][0] + ((I - D) / d) * (points[i][0] - points[i-1][0])
                qy = points[i-1][1] + ((I - D) / d) * (points[i][1] - points[i-1][1])
                qz = points[i-1][2] + ((I - D) / d) * (points[i][2] - points[i-1][2])
                q = [qx, qy, qz]
                newpoints.append(q)
                points.insert(i, q)
                num_newpoints += 1
                D = 0.0
            else:
                D += d
    if len(newpoints) == n - 1:
        temp_p = [points[-1][0], points[-1][0], points[-2][0]]
        newpoints.append(temp_p)
    print(f"Inserted {num_newpoints} new points in total.")
    print(f"Point number after resampling: {len(points)}")

    return points, newpoints # Added new return points



def Scale(points):
    minX, maxX = math.inf, -math.inf
    minY, maxY = math.inf, -math.inf
    minZ, maxZ = math.inf, -math.inf
    for i in range(len(points)):
        minX = min(minX, points[i][0])
        minY = min(minY, points[i][1])
        minZ = min(minZ, points[i][2])
        maxX = max(maxX, points[i][0])
        maxY = max(maxY, points[i][1])
        maxZ = max(maxZ, points[i][2])

    size = max(maxX - minX, maxY - minY, maxZ - minZ)
    newpoints = []
    for i in range(len(points)):
        qx = (points[i][0] - minX) / size *10
        qy = (points[i][1] - minY) / size *10
        qz = (points[i][2] - minZ) / size *10
        newpoints.append([qx, qy, qz])

    return newpoints

def TranslateTo(points, pt):
    c = Centroid(points)
    newpoints = []
    for i in range(len(points)):
        qx = points[i][0] + pt[0] - c[0]
        qy = points[i][1] + pt[1] - c[1]
        qz = points[i][2] + pt[2] - c[2] # Add z-axis
        newpoints.append([qx, qy, qz])

    return newpoints

def Centroid(points): 
    x, y, z = 0.0, 0.0, 0.0
    for i in range(len(points)):
        x += points[i][0]
        y += points[i][1]
        z += points[i][2]
    x /= len(points)
    y /= len(points)
    z /= len(points)

    return [x, y, z]
   
def PathLength(points):
    d = 0.0
    for i in range(1,len(points)):
        d += Distance(points[i-1], points[i])
        
    return d

def Distance(p1, p2):
    d = math.dist(p1, p2)
    return d


def normalizePoints(points, n=14, origin=[0,0,0], vec=[0, -1, 0], rotate=True):
    # full_points, new_points = Resample(points, n)
    # new_points = Scale(new_points)
    # new_points = TranslateTo(new_points, origin)
    full_points = Scale(points)
    full_points = TranslateTo(full_points, origin)
    if rotate:
        full_points = RotateTo(full_points, vec=vec)
    return full_points


def RotateTo(points, vec=[0,-1,0]):
    coordinates = np.array(points)

    # Define the indices of the keypoints that define the body plane
    indices = [2, 5, 8, 11]
    # indices = [1, 11, 3]

    # Extract the coordinates of the keypoints that define the body plane
    plane_points = coordinates[indices]

    # Calculate the normal vector of the body plane using cross product
    vector1 = plane_points[1] - plane_points[0]
    vector2 = plane_points[2] - plane_points[0]
    normal_vector = np.cross(vector1, vector2)

    # Calculate the angle between the normal vector and the predefined vector
    angle = np.arccos(np.dot(normal_vector, vec) / (np.linalg.norm(normal_vector) * np.linalg.norm(vec)))
    # print(f'Rotated {np.degrees(angle)} degrees')

    # Rotate the coordinates along the Z-axis by the calculated angle
    if normal_vector[0] > 0:
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        rotation_matrix = np.array([
            [np.cos(angle), np.sin(angle), 0],
            [-np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

    rotated_coordinates = np.dot(coordinates, rotation_matrix)

    return rotated_coordinates



def cart2polar(points, center, use_degree=False):
    """
    Convert an array of 3D cartesian points to polar coordinates.

    Args:
        points (numpy.ndarray): An Nx3 array of cartesian points.
        center (tuple): A tuple (cx, cy, cz) representing the center.

    Returns:
        numpy.ndarray: An Nx3 array of polar coordinates (r, theta, phi).

    # Example usage:
    center = (0, 0, 0)
    cartesian_points = np.array([[1, 1, 1], [0, 0, 1], [1, 0, 0]])
    polar_coordinates = cart2polar(cartesian_points, center)
    print(polar_coordinates)
    """
    # Calculate the differences between points and the center
    dx = points[:, 0] - center[0]
    dy = points[:, 1] - center[1]
    dz = points[:, 2] - center[2]

    # Calculate r (distance)
    r = np.sqrt(dx**2 + dy**2 + dz**2)

    # Calculate theta (angle with +z axis)
    theta = np.arccos(dz / r)

    # Calculate phi (angle with +x axis)
    phi = np.arctan2(dy, dx)

    # Adjust phi to be in [0, 2*pi)
    phi[phi < 0] += 2 * np.pi

    if use_degree:
        theta = np.rad2deg(theta)
        phi = np.rad2deg(phi)

    return np.column_stack((r, theta, phi))

    






    


  






        