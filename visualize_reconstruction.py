#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def parse_bundle_file(bundle_filename):
    """
    Parses a bundle file with the following format:
      - Comments (lines starting with #) are skipped.
      - Header line: <n_cameras> <n_points>
      - For each camera, three lines:
            1) focal k1 k2
            2) 9 numbers (rotation matrix in row-major order)
            3) 3 numbers (translation vector)
      - For each point: one line with "x y z r g b"
    """
    with open(bundle_filename, 'r') as f:
        lines = f.readlines()

    idx = 0
    while lines[idx].strip().startswith('#'):
        idx += 1
    header_line = lines[idx].strip()
    idx += 1
    n_cameras, n_points = map(int, header_line.split())

    cameras = []
    for i in range(n_cameras):
        # Camera intrinsics (focal, distortion)
        focal_line = lines[idx].strip()
        idx += 1
        focal, k1, k2 = map(float, focal_line.split())
        # Rotation matrix (3x3)
        rot_line = lines[idx].strip()
        idx += 1
        rot_vals = list(map(float, rot_line.split()))
        R = np.array(rot_vals).reshape((3, 3))
        # Translation vector
        trans_line = lines[idx].strip()
        idx += 1
        t = np.array(list(map(float, trans_line.split())))
        cameras.append({'focal': focal, 'R': R, 't': t})

    points = []
    for i in range(n_points):
        line = lines[idx].strip()
        idx += 1
        parts = line.split()
        x, y, z = map(float, parts[0:3])
        r, g, b = map(int, parts[3:6])
        points.append({'coord': np.array([x, y, z]), 'color': (r, g, b)})
    return cameras, points

def parse_ply_file(ply_filename):
    """
    Parses a simple ASCII PLY file. Assumes vertices are given as:
      x y z r g b
    """
    with open(ply_filename, 'r') as f:
        lines = f.readlines()

    num_vertices = 0
    vertex_start = 0
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            num_vertices = int(line.strip().split()[-1])
        if line.strip() == "end_header":
            vertex_start = i + 1
            break

    vertices = []
    for line in lines[vertex_start:vertex_start+num_vertices]:
        parts = line.strip().split()
        x, y, z = map(float, parts[0:3])
        r, g, b = map(int, parts[3:6])
        vertices.append({'coord': np.array([x, y, z]), 'color': (r, g, b)})
    return vertices

def visualize_bundle(cameras, points):
    """
    Visualizes the reconstruction from bundle.out in 3D.
    Computes camera centers from the camera pose (center = -R.T * t)
    and returns the camera centers and point coordinates for further projection.
    """
    camera_centers = []
    for cam in cameras:
        R = cam['R']
        t = cam['t']
        center = -R.T @ t
        camera_centers.append(center)
    camera_centers = np.array(camera_centers)
    points_coords = np.array([p['coord'] for p in points])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_coords[:,0], points_coords[:,1], points_coords[:,2],
               c='k', marker='.', s=1, label='3D Points')
    ax.scatter(camera_centers[:,0], camera_centers[:,1], camera_centers[:,2],
               c='r', marker='o', s=50, label='Cameras')
    ax.set_title('3D Reconstruction from bundle.out')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
    return camera_centers, points_coords

def visualize_ply(vertices):
    """
    Visualizes the reconstruction from reconstruction.ply in 3D.
    Separates vertices by color (cameras in red and points in other colors)
    and returns the camera and point coordinates.
    """
    cameras = []
    points = []
    for v in vertices:
        if v['color'] == (255, 0, 0):
            cameras.append(v['coord'])
        else:
            points.append(v['coord'])
    cameras = np.array(cameras) if cameras else np.empty((0, 3))
    points = np.array(points) if points else np.empty((0, 3))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    if points.size > 0:
        ax.scatter(points[:,0], points[:,1], points[:,2],
                   c='k', marker='.', s=1, label='3D Points')
    if cameras.size > 0:
        ax.scatter(cameras[:,0], cameras[:,1], cameras[:,2],
                   c='r', marker='o', s=50, label='Cameras')
    ax.set_title('3D Reconstruction from reconstruction.ply')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
    return cameras, points

def visualize_projections(camera_centers, points_coords):
    """
    Creates a new figure with three subplots showing 2D projections along:
      - XY (X vs. Y)
      - XZ (X vs. Z)
      - YZ (Y vs. Z)
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # XY projection
    axs[0].scatter(points_coords[:,0], points_coords[:,1], c='k', s=1, label='Points')
    axs[0].scatter(camera_centers[:,0], camera_centers[:,1], c='r', s=50, label='Cameras')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].set_title('XY Projection')
    axs[0].legend()
    
    # XZ projection
    axs[1].scatter(points_coords[:,0], points_coords[:,2], c='k', s=1, label='Points')
    axs[1].scatter(camera_centers[:,0], camera_centers[:,2], c='r', s=50, label='Cameras')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Z')
    axs[1].set_title('XZ Projection')
    axs[1].legend()
    
    # YZ projection
    axs[2].scatter(points_coords[:,1], points_coords[:,2], c='k', s=1, label='Points')
    axs[2].scatter(camera_centers[:,1], camera_centers[:,2], c='r', s=50, label='Cameras')
    axs[2].set_xlabel('Y')
    axs[2].set_ylabel('Z')
    axs[2].set_title('YZ Projection')
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Visualize reconstruction using bundle.out and reconstruction.ply files with 2D projections.'
    )
    parser.add_argument('--bundle', type=str, default='bundle.out',
                        help='Path to bundle.out file')
    parser.add_argument('--ply', type=str, default='reconstruction.ply',
                        help='Path to reconstruction.ply file')
    parser.add_argument('--mode', type=str, choices=['bundle', 'ply', 'both'],
                        default='both', help='Which file(s) to visualize')
    args = parser.parse_args()

    if args.mode in ['bundle', 'both']:
        print("Visualizing bundle.out ...")
        cameras, points = parse_bundle_file(args.bundle)
        cam_centers, pts_coords = visualize_bundle(cameras, points)
        print("Visualizing 2D projections from bundle.out ...")
        visualize_projections(cam_centers, pts_coords)

    if args.mode in ['ply', 'both']:
        print("Visualizing reconstruction.ply ...")
        vertices = parse_ply_file(args.ply)
        cam_coords, pts_coords = visualize_ply(vertices)
        print("Visualizing 2D projections from reconstruction.ply ...")
        if cam_coords.size > 0 and pts_coords.size > 0:
            visualize_projections(cam_coords, pts_coords)

