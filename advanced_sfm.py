#!/usr/bin/env python3
"""
Advanced Robust SfM Pipeline with Global Bundle Adjustment

This script demonstrates a more advanced SfM pipeline:
  1. It reads all JPEG images from a directory.
  2. For each image, it extracts SIFT features.
  3. Pairwise matches are computed and “tracks” across images are built
     using a union–find (disjoint–set) algorithm.
  4. An initial reconstruction is obtained:
     - A seed pair is reconstructed via the Essential matrix and recoverPose.
     - For subsequent images, PnP is used to estimate camera poses.
     - 3D points are initialized via triangulation from tracks.
  5. A global bundle adjustment is run (using SciPy’s least_squares with a Huber loss)
     to refine all camera poses and 3D points jointly.
  6. Finally, a PLY file is written with camera centers (in red) and 3D points (in white).

Dependencies:
  - OpenCV (with opencv-contrib-python for SIFT)
  - NumPy
  - SciPy
  - Pillow
  - Matplotlib (optional, for debugging/visualization)

Usage:
    python advanced_sfm.py <image_dir> [--focal <focal_pixels>] [--out <output_ply>]

Example:
    python advanced_sfm.py ./notredame_images --focal 1500 --out reconstruction.ply
"""

import os, glob, cv2, random
import numpy as np
from PIL import Image, ExifTags
from scipy.optimize import least_squares
import argparse

# ------------------------------
# 1. Utility Functions
# ------------------------------

def get_focal_length(image_path):
    """Extract focal length (assumed in pixels or convertible) from EXIF if available."""
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if exif is not None:
            for tag, value in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if decoded == 'FocalLength':
                    if isinstance(value, tuple) and len(value) == 2:
                        return float(value[0]) / float(value[1])
                    else:
                        return float(value)
    except Exception as e:
        print(f"Error reading EXIF from {image_path}: {e}")
    return None

def build_intrinsics(image_path, focal_override=None):
    """Build a 3x3 intrinsic matrix from image size and focal length (in pixels).
       If a focal_override is given, that value is used."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    if focal_override is not None:
        f = focal_override
    else:
        f_exif = get_focal_length(image_path)
        if f_exif is not None:
            # (Here we assume EXIF value is already in pixel units or roughly so.)
            f = f_exif
        else:
            f = 0.9 * w  # rough guess if no EXIF
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0,  1]], dtype=np.float64)
    return K, (w, h)

def compute_sift_features(image_path):
    """Detect SIFT keypoints and descriptors."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot load image: {image_path}")
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def match_features(desc1, desc2, ratio_thresh=0.75):
    """Match descriptors using FLANN and apply Lowe’s ratio test."""
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in knn if m.distance < ratio_thresh * n.distance]
    return good

# ------------------------------
# 2. Track Building via Union-Find
# ------------------------------

class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]
    def union(self, item1, item2):
        r1 = self.find(item1)
        r2 = self.find(item2)
        if r1 != r2:
            self.parent[r2] = r1
    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item

def build_tracks(image_features, pairwise_matches, min_track_length=2):
    """
    Build tracks (groups of corresponding features across images).
    Each feature is identified as (image_index, keypoint_index, (x,y)).
    Only tracks that appear in at least min_track_length distinct images are kept.
    """
    uf = UnionFind()
    # Add all features.
    for img_idx, (kps, _) in image_features.items():
        for kp_idx in range(len(kps)):
            uf.add((img_idx, kp_idx))
    # Union matched features.
    for (i, j, matches) in pairwise_matches:
        for m in matches:
            uf.union((i, m.queryIdx), (j, m.trainIdx))
    # Group features by root.
    tracks = {}
    for img_idx, (kps, _) in image_features.items():
        for kp_idx, kp in enumerate(kps):
            root = uf.find((img_idx, kp_idx))
            if root not in tracks:
                tracks[root] = []
            tracks[root].append((img_idx, kp_idx, kp.pt))
    # Keep tracks seen in at least min_track_length distinct images.
    tracks = {k: v for k, v in tracks.items() if len(set(obs[0] for obs in v)) >= min_track_length}
    return tracks

# ------------------------------
# 3. Triangulation Helper
# ------------------------------

def triangulate_point(obs1, obs2, K, R1, t1, R2, t2):
    """Triangulate a 3D point from two observations (each a (u,v) tuple)."""
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    pts1 = np.array(obs1).reshape(2, 1)
    pts2 = np.array(obs2).reshape(2, 1)
    pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    pts3d = (pts4d / pts4d[3])[:3].ravel()
    return pts3d

# ------------------------------
# 4. Global Bundle Adjustment (BA)
# ------------------------------

def project_point(camera_params, point_3d, K):
    """
    Project a 3D point into a camera given its 6-parameter pose (rvec, tvec).
    A safeguard is added to avoid division by zero.
    """
    rvec = camera_params[:3]
    tvec = camera_params[3:6]
    R, _ = cv2.Rodrigues(rvec)
    p_cam = R @ point_3d + tvec
    p_proj = K @ p_cam
    # Clip the depth to avoid division by zero.
    z = p_proj[2] if p_proj[2] > 1e-6 else 1e-6
    return p_proj[:2] / z


def ba_residuals(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    """Compute reprojection residuals for BA.
       - params: concatenated camera parameters (6 per camera) and 3D points (3 per point).
       - camera_indices, point_indices, points_2d: observation info.
    """
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    residuals = []
    for i in range(len(points_2d)):
        cam_idx = camera_indices[i]
        pt_idx = point_indices[i]
        proj = project_point(camera_params[cam_idx], points_3d[pt_idx], K)
        residuals.extend(points_2d[i] - proj)
    return np.array(residuals)

def global_bundle_adjustment(camera_params, points_3d, observations, K):
    """
    Run global bundle adjustment.
    observations: list of tuples (cam_idx, point_idx, [u, v])
    Returns optimized camera_params and points_3d.
    """
    camera_indices = []
    point_indices = []
    points_2d = []
    for (cam_idx, pt_idx, meas) in observations:
        camera_indices.append(cam_idx)
        point_indices.append(pt_idx)
        points_2d.append(np.array(meas))
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    points_2d = np.array(points_2d)
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    res = least_squares(ba_residuals, x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K),
                        loss='huber')
    optimized_camera_params = res.x[:n_cameras * 6].reshape((n_cameras, 6))
    optimized_points_3d = res.x[n_cameras * 6:].reshape((n_points, 3))
    return optimized_camera_params, optimized_points_3d

# ------------------------------
# 5. Advanced SfM Pipeline
# ------------------------------

def advanced_sfm(image_dir, focal_override=None):
    """
    Advanced incremental SfM pipeline with global BA.
      - Extract features from all images.
      - Compute pairwise matches and build tracks.
      - Estimate initial camera poses:
           • Use a seed pair (first valid pair) for essential matrix recovery.
           • For other images, use PnP with 2D-3D correspondences from tracks.
      - For each track (seen in at least two images with known poses),
        triangulate an initial 3D point.
      - Build a list of observations (camera index, point index, 2D measurement)
        from the tracks.
      - Run global BA to refine all camera poses and 3D points.
    Returns:
      - camera_poses: dict mapping image index → (R, t)
      - optimized 3D points (points3D)
      - intrinsics (assumed similar for all images)
      - list of image paths.
    """
    # 5.1 Load images and extract features
    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if len(img_paths) < 2:
        raise ValueError("Need at least 2 images for reconstruction.")
    image_features = {}
    intrinsics = {}
    for i, path in enumerate(img_paths):
        K, _ = build_intrinsics(path, focal_override)
        intrinsics[i] = K
        kps, desc = compute_sift_features(path)
        image_features[i] = (kps, desc)
    print(f"Extracted features from {len(img_paths)} images.")

    # 5.2 Compute pairwise matches (here we try all pairs for robustness)
    pairwise_matches = []
    n_images = len(img_paths)
    for i in range(n_images):
        for j in range(i+1, n_images):
            kps1, desc1 = image_features[i]
            kps2, desc2 = image_features[j]
            if desc1 is None or desc2 is None:
                continue
            matches = match_features(desc1, desc2)
            if len(matches) > 15:
                pairwise_matches.append((i, j, matches))
    print(f"Found {len(pairwise_matches)} pairwise matches.")

    # 5.3 Build tracks across images.
    tracks = build_tracks(image_features, pairwise_matches, min_track_length=2)
    print(f"Built {len(tracks)} tracks.")

    # 5.4 Initial camera pose estimation.
    # Use first valid pair as seed.
    seed = pairwise_matches[0]
    i0, i1, matches_seed = seed
    kps0, _ = image_features[i0]
    kps1, _ = image_features[i1]
    K0 = intrinsics[i0]
    pts0 = np.float32([kps0[m.queryIdx].pt for m in matches_seed])
    pts1 = np.float32([kps1[m.trainIdx].pt for m in matches_seed])
    E, mask = cv2.findEssentialMat(pts0, pts1, K0, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R1, t1, _ = cv2.recoverPose(E, pts0, pts1, K0)
    camera_poses = {}
    camera_poses[i0] = (np.eye(3), np.zeros((3,1)))
    camera_poses[i1] = (R1, t1)
    print(f"Seed pair: camera {i0} (identity) and camera {i1} (recovered pose).")

    # For remaining images, use PnP on tracks that include already reconstructed cameras.
    for i in range(n_images):
        if i in camera_poses:
            continue
        pts_3d = []
        pts_2d = []
        for track in tracks.values():
            # Check if track is seen in a camera with known pose and in image i.
            cams_in_track = [obs[0] for obs in track]
            if (any(cam in camera_poses for cam in cams_in_track)) and (i in cams_in_track):
                # Use the observation from camera with known pose to get an initial 3D point.
                for obs in track:
                    if obs[0] in camera_poses:
                        # We triangulate from the first two available views in the track.
                        if len(track) >= 2:
                            obs1 = track[0][2]  # (u,v)
                            obs2 = track[1][2]
                            cam1 = track[0][0]
                            cam2 = track[1][0]
                            if cam1 in camera_poses and cam2 in camera_poses:
                                R_a, t_a = camera_poses[cam1]
                                R_b, t_b = camera_poses[cam2]
                                K_a = intrinsics[cam1]
                                pt3d = triangulate_point(obs1, obs2, K_a, R_a, t_a, R_b, t_b)
                                pts_3d.append(pt3d)
                                # And take the 2D measurement from image i.
                                for obs_i in track:
                                    if obs_i[0] == i:
                                        pts_2d.append(obs_i[2])
                                        break
                        break
        pts_3d = np.array(pts_3d, dtype=np.float32)
        pts_2d = np.array(pts_2d, dtype=np.float32)
        if len(pts_3d) < 6:
            print(f"Not enough points to estimate pose for camera {i}, skipping.")
            continue
        K_i = intrinsics[i]
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, K_i, None)
        if not retval:
            continue
        R_i, _ = cv2.Rodrigues(rvec)
        camera_poses[i] = (R_i, tvec)
        print(f"Estimated pose for camera {i} with {len(inliers)} inliers.")

    # 5.5 Triangulate 3D points from tracks.
    points3D = []
    track_to_point = {}  # map track id to point index
    observations = []    # list of (camera_index, point_index, measurement)
    point_idx = 0
    for track_id, track in tracks.items():
        # Only use tracks seen in at least 2 cameras with known pose.
        valid_obs = [obs for obs in track if obs[0] in camera_poses]
        if len(valid_obs) < 2:
            continue
        # Use the first two valid observations to triangulate.
        cam_idx1, kp_idx1, pt1 = valid_obs[0]
        cam_idx2, kp_idx2, pt2 = valid_obs[1]
        R1, t1 = camera_poses[cam_idx1]
        R2, t2 = camera_poses[cam_idx2]
        K1 = intrinsics[cam_idx1]
        pt3d = triangulate_point(pt1, pt2, K1, R1, t1, R2, t2)
        points3D.append(pt3d)
        track_to_point[track_id] = point_idx
        # Record every observation in the track (for BA).
        for obs in valid_obs:
            observations.append((obs[0], point_idx, obs[2]))
        point_idx += 1
    points3D = np.array(points3D)
    print(f"Triangulated {points3D.shape[0]} 3D points from tracks.")

    # 5.6 Prepare camera parameters for BA.
    n_cameras = max(camera_poses.keys()) + 1
    camera_params = np.zeros((n_cameras, 6), dtype=np.float64)
    for i in range(n_cameras):
        if i in camera_poses:
            R, t = camera_poses[i]
            rvec, _ = cv2.Rodrigues(R)
            camera_params[i, :3] = rvec.ravel()
            camera_params[i, 3:6] = t.ravel()
        else:
            # Leave cameras without an estimated pose as zeros.
            pass

    # 5.7 Run global bundle adjustment.
    print("Running global bundle adjustment...")
    optimized_camera_params, optimized_points3D = global_bundle_adjustment(camera_params, points3D, observations, intrinsics[0])
    print("Bundle adjustment complete.")

    # Update camera_poses from optimized parameters.
    for i in range(n_cameras):
        rvec = optimized_camera_params[i, :3]
        tvec = optimized_camera_params[i, 3:6]
        R, _ = cv2.Rodrigues(rvec)
        camera_poses[i] = (R, tvec.reshape((3,1)))

    return camera_poses, optimized_points3D, intrinsics, img_paths

# ------------------------------
# 6. Write PLY File
# ------------------------------

def write_ply(filename, camera_poses, points3D):
    """Write a simple PLY file with camera centers (in red) and 3D points (in white)."""
    with open(filename, 'w') as f:
        num_vertices = len(camera_poses) + len(points3D)
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        # Write camera centers (in red)
        for i in range(len(camera_poses)):
            R, t = camera_poses[i]
            c = -R.T @ t
            f.write(f"{c[0,0]} {c[1,0]} {c[2,0]} 255 0 0\n")
        # Write 3D points (in white)
        for p in points3D:
            f.write(f"{p[0]} {p[1]} {p[2]} 255 255 255\n")
    print(f"PLY file written to {filename}")

# ------------------------------
# 7. Main
# ------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced robust SfM pipeline with global bundle adjustment.")
    parser.add_argument("image_dir", help="Directory containing JPEG images.")
    parser.add_argument("--focal", type=float, default=None, help="Focal length override in pixels.")
    parser.add_argument("--out", type=str, default="advanced_reconstruction.ply", help="Output PLY filename.")
    args = parser.parse_args()

    print("Running advanced SfM pipeline on images in:", args.image_dir)
    cam_poses, pts3D, intrinsics, img_paths = advanced_sfm(args.image_dir, focal_override=args.focal)
    write_ply(args.out, cam_poses, pts3D)
