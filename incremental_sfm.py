#!/usr/bin/env python3

import os
import cv2
import numpy as np
from PIL import Image, ExifTags
import glob

# ----------------------------------------------------
# 1) Utility functions
# ----------------------------------------------------

def get_focal_length(image_path):
    """
    Extracts the focal length (in pixels) from EXIF data if available.
    Returns None if unavailable.
    """
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if exif is not None:
            for tag, value in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if decoded == 'FocalLength':
                    # value can be (num, den)
                    if isinstance(value, tuple) and len(value) == 2:
                        return float(value[0]) / float(value[1])
                    else:
                        return float(value)
    except:
        pass
    return None

def build_intrinsics(image_path, focal_override=None):
    """
    Builds a 3x3 camera intrinsic matrix using:
    - focal length from EXIF (in mm) and approximate sensor width,
      or a user-provided override (in pixels).
    - principal point at the image center.
    In a real pipeline, you'd also handle sensor size to convert mm -> pixels properly.
    """
    # Load image just to get shape
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]

    # Approx: if we have an EXIF focal in mm, we typically need to convert to pixel units:
    # focal_pixels = (f_mm / sensor_width_mm) * image_width_in_pixels
    # For simplicity, we just take the user override or fallback to a fixed guess
    if focal_override is not None:
        f = focal_override
    else:
        f_exif = get_focal_length(image_path)
        if f_exif is not None:
            # This is usually in mm, but let's pretend it's in pixels for simplicity
            # Real code should convert mm -> pixels if sensor size is known.
            f = f_exif
        else:
            # fallback if EXIF not found
            f = 0.9 * w  # just a rough guess

    cx = w / 2.0
    cy = h / 2.0

    K = np.array([
        [f,   0,  cx],
        [0,   f,  cy],
        [0,   0,   1]
    ], dtype=np.float64)
    return K, (w, h)

def extract_features(image_path):
    """
    Extracts SIFT keypoints and descriptors from an image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot load image: {image_path}")
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def match_features(desc1, desc2, ratio_thresh=0.75):
    """
    Matches two sets of SIFT descriptors using FLANN + Lowe's ratio test.
    Returns a list of good matches.
    """
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(desc1, desc2, k=2)

    good = []
    for m, n in knn:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    return good

def pose_from_essential(kps1, kps2, K, matches):
    """
    Given matched keypoints in two images and the camera intrinsics K,
    estimate the essential matrix (via RANSAC) and recover the relative pose (R, t).
    Returns R, t, and inlier matches.
    """
    # Convert keypoints to numpy arrays of shape (N, 2)
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

    # Normalize by K if desired, or just use findEssentialMat with K
    E, mask = cv2.findEssentialMat(pts1, pts2, K,
                                   method=cv2.RANSAC, prob=0.999, threshold=1.0)
    # mask is an inlier/outlier mask for each match
    inliers = mask.ravel().tolist()

    # Recover pose from E
    # Note: returns 3x3 R, 3x1 t
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    inliers_pose = mask_pose.ravel().tolist()

    # We'll combine them so only matches that are inliers in the final pose remain
    # but in practice, findEssentialMat's mask and recoverPose's mask often coincide
    inlier_matches = []
    for match_i, (m, inlE, inlP) in enumerate(zip(matches, inliers, inliers_pose)):
        if inlE and inlP:
            inlier_matches.append(m)

    return R, t, inlier_matches

def triangulate_points(kps1, kps2, K, R1, t1, R2, t2, matches):
    """
    Triangulate 3D points given matched keypoints between two views (R1,t1) and (R2,t2).
    Returns Nx3 array of 3D points in the coordinate system of the first camera.
    """
    # Build projection matrices P1, P2 in homogeneous coords
    #   P = K * [R|t]
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))

    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

    # Triangulate in homogeneous coordinates
    pts4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

    # Convert from homogeneous to 3D
    pts4d = (pts4d_hom / pts4d_hom[3])[:3].T  # shape (N,3)
    return pts4d

def get_camera_center(R, t):
    """
    Given a camera pose [R|t], the center C is -R^T t in world coordinates.
    """
    return -R.T @ t

# ----------------------------------------------------
# 2) Main incremental pipeline
# ----------------------------------------------------
def incremental_sfm(image_dir, focal_override=None):
    """
    A minimal, incremental SfM pipeline:
      - Reads all images
      - For each consecutive pair, we estimate pose, triangulate points
      - For subsequent images, we do PnP to get the new pose, then triangulate new points
    NOTE: No loop closure or full BA, so results are likely approximate.
    """

    # 1. Gather images
    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if len(img_paths) < 2:
        raise ValueError("Need at least two images for reconstruction.")

    # 2. Build intrinsics and extract features for each image
    all_K = []
    all_size = []
    all_keypoints = []
    all_descriptors = []
    for path in img_paths:
        K, size = build_intrinsics(path, focal_override)
        all_K.append(K)
        all_size.append(size)
        kps, desc = extract_features(path)
        all_keypoints.append(kps)
        all_descriptors.append(desc)

    # We'll assume all images share the same intrinsics for simplicity
    # (If each image has different focal length from EXIF, you could store them separately.)
    K = all_K[0]

    # 3. Initialize with the first two images
    matches_01 = match_features(all_descriptors[0], all_descriptors[1])
    R1 = np.eye(3, dtype=np.float64)
    t1 = np.zeros((3,1), dtype=np.float64)  # world origin for first camera

    R2, t2, inliers_01 = pose_from_essential(
        all_keypoints[0], all_keypoints[1], K, matches_01
    )

    # Triangulate points between first two images
    pts3D_01 = triangulate_points(
        all_keypoints[0], all_keypoints[1], K,
        R1, t1, R2, t2, inliers_01
    )

    # Store camera poses
    camera_poses = [(R1, t1), (R2, t2)]
    # We'll store 3D points in a list (no merging across multiple views in this minimal example)
    global_points3D = pts3D_01.tolist()

    # 4. For each subsequent image i, match to the previous image (i-1)
    #    do solvePnP to estimate R,t, then triangulate new points
    for i in range(2, len(img_paths)):
        desc_i = all_descriptors[i]
        kps_i = all_keypoints[i]

        # Match with the previous image
        desc_prev = all_descriptors[i - 1]
        kps_prev = all_keypoints[i - 1]
        matches_i = match_features(desc_prev, desc_i)

        # Convert matches to 2D-3D pairs by reprojecting the 3D points
        # (In a more robust approach, you'd track which 2D features correspond to 3D from earlier triangulation.
        #  Here we only do a naive approach using the last frame's newly triangulated points.)

        # 4a. We'll use the last camera's 3D points (which might be partial).
        # This is a big simplification; real SfM merges all previous points.
        R_prev, t_prev = camera_poses[i - 1]

        # Triangulate between the last two images
        # But we also need 2D->3D correspondences to run PnP.
        # A minimal approach: we just triangulate first, then do PnP with the new 3D points.
        # But that is somewhat circular. The robust approach is to use existing 3D from the map,
        # matched to the new image's 2D keypoints. Let's do a naive approach:

        # We'll just do PnP with the 2D->3D from the previous step's triangulated points:
        # That requires we know which matches correspond to those 3D points.
        # Since we do not track them carefully, let's do a small hack:
        #   - Re-triangulate between (i-1) and (i) with the initial guess R_prev, t_prev = identity for the new camera,
        #     then refine pose with PnP. This is not a standard pipeline, but a demonstration.

        # Step 1: Triangulate with guess R_i=R_prev, t_i = t_prev (i.e. same pose as the last camera).
        #         Then we get some approximate 3D. Then we do PnP to refine R_i,t_i.
        R_guess = R_prev.copy()
        t_guess = t_prev.copy()

        # We do an approximate triangulation
        pts3D_approx = triangulate_points(kps_prev, kps_i, K, R_prev, t_prev, R_guess, t_guess, matches_i)

        # Step 2: Now we have a set of 3D points in the coordinate frame of the (i-1)th camera.
        # Build 2D->3D correspondences
        pts2 = []
        pts3 = []
        for m, X in zip(matches_i, pts3D_approx):
            # 2D point in image i
            uv = kps_i[m.trainIdx].pt
            pts2.append(uv)
            # 3D point in (i-1)th camera coordinates
            pts3.append(X)

        pts2 = np.array(pts2, dtype=np.float32)
        pts3 = np.array(pts3, dtype=np.float32)

        # Step 3: Solve PnP to refine R,t
        # We need to transform 3D points from the (i-1) camera frame into the first camera's frame (world frame).
        # The (i-1)th camera is at R_prev,t_prev in the world. So a 3D point in (i-1)th frame is
        #   X_world = R_prev * X + t_prev
        # Let's do that:
        pts3_world = (R_prev @ pts3.T).T + t_prev.ravel()

        # Now run PnP
        # We'll use the default OpenCV solver with RANSAC. This returns rvec, tvec in the world frame.
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3_world, pts2, K, distCoeffs=None,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not retval:
            print(f"Pose estimation failed for image {i}. Using last pose as fallback.")
            R_i, t_i = R_prev, t_prev
        else:
            # Convert rvec->R
            R_i, _ = cv2.Rodrigues(rvec)
            t_i = tvec

        # Store the new camera pose
        camera_poses.append((R_i, t_i))

        # Triangulate new points between (i-1)th and i-th camera using the refined pose
        pts3D_new = triangulate_points(kps_prev, kps_i, K, R_prev, t_prev, R_i, t_i, matches_i)

        # Convert to world coordinates
        pts3D_new_world = (R_prev @ pts3D_new.T).T + t_prev.ravel()

        # Accumulate in the global structure
        global_points3D.extend(pts3D_new_world.tolist())

    # ------------------------------------------------
    # Build final lists for cameras + points
    # ------------------------------------------------
    # Camera centers in world coordinates
    camera_centers = []
    for (R, t) in camera_poses:
        c = get_camera_center(R, t)
        camera_centers.append(c.ravel().tolist())

    points3D = np.array(global_points3D)
    return camera_centers, points3D

# ----------------------------------------------------
# 3) Write output to a simple PLY file
# ----------------------------------------------------
def write_ply(filename, camera_centers, points3D):
    """
    Writes a PLY file with camera centers in red and points in white.
    """
    with open(filename, 'w') as f:
        # header
        num_vertices = len(camera_centers) + len(points3D)
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

        # camera centers in red
        for c in camera_centers:
            f.write(f"{c[0]} {c[1]} {c[2]} 255 0 0\n")

        # 3D points in white
        for p in points3D:
            f.write(f"{p[0]} {p[1]} {p[2]} 255 255 255\n")
# ----------------------------------------------------
# 4) Main
# ----------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Minimal incremental SfM with OpenCV (no BA, no loop closure)."
    )
    parser.add_argument("image_dir", help="Path to directory of images (JPEG).")
    parser.add_argument("--focal", type=float, default=None,
                        help="Override focal length in pixels (if known).")
    parser.add_argument("--out", type=str, default="reconstruction.ply",
                        help="Output PLY file name.")
    args = parser.parse_args()

    # Run incremental SfM
    print("Running incremental SfM on images in:", args.image_dir)
    cam_centers, pts3D = incremental_sfm(args.image_dir, focal_override=args.focal)

    # Write results to PLY
    print(f"Writing {len(pts3D)} points and {len(cam_centers)} cameras to {args.out}")
    write_ply(args.out, cam_centers, pts3D)
    print("Done.")


