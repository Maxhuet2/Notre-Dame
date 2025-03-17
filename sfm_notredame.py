#!/usr/bin/env python3
import os
import cv2
import numpy as np
import random
from PIL import Image, ExifTags
import matplotlib.pyplot as plt

# ------------------------------
# 1) Détection de Notre-Dame via SIFT + Homographie
# ------------------------------
#def detect_notredame_in_image(test_image_path, ref_image_path, ratio_thresh=0.75, min_inliers=30):
#    img_test = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
#    img_ref = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)
#    if img_test is None or img_ref is None:
#        print(f"Impossible de lire {test_image_path} ou {ref_image_path}")
#        return False

#    sift = cv2.SIFT_create()
#    kp_test, desc_test = sift.detectAndCompute(img_test, None)
#    kp_ref, desc_ref = sift.detectAndCompute(img_ref, None)

#   if desc_test is None or desc_ref is None:
#        print("Descripteurs vides")
#        return False

#    index_params = dict(algorithm=1, trees=5)
#    search_params = dict(checks=50)
#    flann = cv2.FlannBasedMatcher(index_params, search_params)
#    matches_knn = flann.knnMatch(desc_ref, desc_test, k=2)

#    good_matches = []
#    for m, n in matches_knn:
#        if m.distance < ratio_thresh * n.distance:
#            good_matches.append(m)

#    if len(good_matches) < min_inliers:
#        return False

#    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#    dst_pts = np.float32([kp_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#    if H is None:
#        return False

#    inliers = np.sum(mask)
#    if inliers >= min_inliers:
#        print(f"-> Notre-Dame détectée dans {test_image_path} (inliers={inliers})")
#        return True
#    else:
#        return False

# ------------------------------
# 2) Extraction de la focale via EXIF
# ------------------------------
def get_focal_length(image_path):
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

# ------------------------------
# 3) SIFT Feature Extraction
# ------------------------------
def compute_sift_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# ------------------------------
# 4) FLANN-based matching (utilisé dans two_view_triangulation)
# ------------------------------
def match_features_with_stats(desc1, desc2, ratio_thresh=0.5):
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches_knn = flann.knnMatch(desc1, desc2, k=2)
    raw_matches = len(matches_knn)
    good_matches = []
    for m, n in matches_knn:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    return raw_matches, good_matches

# ------------------------------
# 5) Triangulation sur deux vues avec collecte de statistiques
# ------------------------------
def two_view_triangulation(
    img_path1, img_path2,
    kps1, desc1,
    kps2, desc2,
    focal1, focal2,
    ratio_thresh=0.9
):
    # Lecture des images couleur pour dimensions
    img1_color = cv2.imread(img_path1)
    img2_color = cv2.imread(img_path2)
    if img1_color is None or img2_color is None:
        print("Erreur lecture images pour triangulation")
        return None, None, None, None, None

    h1, w1 = img1_color.shape[:2]
    h2, w2 = img2_color.shape[:2]

    f1 = focal1 if focal1 else 800.0
    f2 = focal2 if focal2 else 800.0
    cx1, cy1 = w1/2.0, h1/2.0
    cx2, cy2 = w2/2.0, h2/2.0

    K1 = np.array([[f1, 0, cx1],
                   [0, f1, cy1],
                   [0,  0,   1]], dtype=np.float64)
    K2 = np.array([[f2, 0, cx2],
                   [0, f2, cy2],
                   [0,  0,   1]], dtype=np.float64)

    # Matching avec statistiques
    raw_matches, good_matches = match_features_with_stats(desc1, desc2, ratio_thresh=ratio_thresh)
    if len(good_matches) < 100:
        print("Trop peu de matches pour trianguler.")
        metrics = {
            "raw_matches": raw_matches,
            "good_matches": len(good_matches),
            "inliers_F": 0,
            "inliers_pose": 0
        }
        return None, None, metrics, None, None

    pts1 = np.float32([kps1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good_matches])

    F, mask_F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1, 0.99)
    if F is None or mask_F is None:
        print("Erreur dans F")
        metrics = {
            "raw_matches": raw_matches,
            "good_matches": len(good_matches),
            "inliers_F": 0,
            "inliers_pose": 0
        }
        return None, None, metrics, None, None
    
    inliers_F = int(np.sum(mask_F))
    
    E = K2.T @ F @ K1

    pts1_reshaped = pts1.reshape(-1, 1, 2)
    pts2_reshaped = pts2.reshape(-1, 1, 2)
    retval, R, t, mask_pose = cv2.recoverPose(E, pts1_reshaped, pts2_reshaped, K1)
    print("mask_pose shape:", mask_pose.shape, "sum:", np.sum(mask_pose))

    # Si on est sûr que c’est 255 pour inlier
    inliers_pose = int(np.sum(mask_pose) / 255) if mask_pose is not None else 0

    if retval < 50:
        print("Peu d'inliers dans recoverPose.")
        metrics = {
            "raw_matches": raw_matches,
            "good_matches": len(good_matches),
            "inliers_F": inliers_F,
            "inliers_pose": inliers_pose
        }
        return None, None, metrics, None, None

    R = R.astype(np.float64)
    t = t.astype(np.float64)

    proj1 = K1 @ np.hstack((np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)))
    proj2 = K2 @ np.hstack((R, t))

# Au lieu de : inlier_idx = np.where(mask_pose.ravel() == 1)[0]
# Faire :
#inlier_idx = np.where(mask_pose.ravel() != 0)[0]

# si c’est strictement 255 :
    inlier_idx = np.where(mask_pose.ravel() == 255)[0]


    if len(inlier_idx) == 0:
        print("Aucun inlier pour triangulation.")
        metrics = {
            "raw_matches": raw_matches,
            "good_matches": len(good_matches),
            "inliers_F": inliers_F,
            "inliers_pose": inliers_pose
        }
        return None, None, metrics, None, None

    pts1_in = pts1[inlier_idx].T.astype(np.float64)
    pts2_in = pts2[inlier_idx].T.astype(np.float64)
    proj1 = np.ascontiguousarray(proj1)
    proj2 = np.ascontiguousarray(proj2)
    pts1_in = np.ascontiguousarray(pts1_in)
    pts2_in = np.ascontiguousarray(pts2_in)
    pts4D = cv2.triangulatePoints(proj1, proj2, pts1_in, pts2_in)
    pts4D /= pts4D[3]
    points3D = pts4D[:3].T

    pose1 = np.eye(4, dtype=np.float64)
    pose2 = np.eye(4, dtype=np.float64)
    pose2[:3, :3] = R
    pose2[:3, 3] = t.ravel()

    camera_poses = {0: pose1, 1: pose2}
    inlier_matches = [good_matches[i] for i in inlier_idx]

    metrics = {
        "raw_matches": raw_matches,
        "good_matches": len(good_matches),
        "inliers_F": inliers_F,
        "inliers_pose": inliers_pose
    }
    return camera_poses, points3D, metrics, good_matches, inlier_matches

# ------------------------------
# 6) Visualisation des appariements
# ------------------------------
def visualize_matches(img_path1, img_path2, kps1, kps2, matches, window_name="Matches"):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    if img1 is None or img2 is None:
        print("Erreur lecture images pour visualisation")
        return
    drawn = cv2.drawMatches(img1, kps1, img2, kps2, matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow(window_name, drawn)
    print("Appuyez sur une touche...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Pour sauvegarder: cv2.imwrite("matches.jpg", drawn)

# ------------------------------
# 7) Écriture des résultats
# ------------------------------
def write_ply_file(ply_path, camera_poses, points3D):
    with open(ply_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        num_vertices = len(points3D) + len(camera_poses)
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pose in camera_poses.values():
            R = pose[:3, :3]
            t = pose[:3, 3]
            cam_center = -R.T @ t
            f.write(f"{cam_center[0]} {cam_center[1]} {cam_center[2]} 255 0 0\n")
        for pt in points3D:
            f.write(f"{pt[0]} {pt[1]} {pt[2]} 255 255 255\n")
    print(f"PLY file écrit dans {ply_path}")

def write_bundle_file(bundle_path, camera_poses, points3D):
    with open(bundle_path, 'w') as f:
        f.write("# Simple 2-view bundle file\n")
        f.write(f"{len(camera_poses)} {len(points3D)}\n")
        for cam_idx in sorted(camera_poses.keys()):
            f.write("800 0.0 0.0\n")
            pose = camera_poses[cam_idx]
            R = pose[:3, :3].flatten()
            t = pose[:3, 3]
            f.write(" ".join(map(str, R)) + "\n")
            f.write(" ".join(map(str, t)) + "\n")
        for pt in points3D:
            f.write(" ".join(map(str, pt)) + " 255 255 255\n")
    print(f"Bundle file écrit dans {bundle_path}")

# ------------------------------
# 8) Pipeline principal et collecte de statistiques
# ------------------------------
def main(image_dir, ref_path="notredame_reference.jpg"):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    image_files.sort()
    if len(image_files) > 30:
        random.seed(39)
        image_files = random.sample(image_files, 10)
        image_files.sort()

    print("=== Détection de Notre-Dame dans les images ===")
    images_with_nd = []
    for idx, fname in enumerate(image_files):
        path = os.path.join(image_dir, fname)
        images_with_nd.append((idx,fname))
        #if detect_notredame_in_image(path, ref_path):
            #images_with_nd.append((idx, fname))

    if len(images_with_nd) < 2:
        print("Pas assez d'images contenant ND pour trianguler.")
        return

    # Extraction des features et focales
    image_features = {}
    focal_lengths = {}
    for idx, fname in enumerate(image_files):
        path = os.path.join(image_dir, fname)
        focal = get_focal_length(path)
        f = focal if focal is not None else 800.0
        kps, desc = compute_sift_features(path)
        image_features[idx] = (kps, desc)
        focal_lengths[idx] = f

    pair_stats = []  # liste pour stocker les stats de chaque paire
    successful_pairs = []  # pour les paires ayant une triangulation réussie

    # Boucle sur les paires d'images contenant ND
    for i in range(len(images_with_nd)):
        for j in range(i+1, len(images_with_nd)):
            idx1, fname1 = images_with_nd[i]
            idx2, fname2 = images_with_nd[j]
            print(f"--- Tentative sur la paire : {fname1} et {fname2} ---")
            path1 = os.path.join(image_dir, fname1)
            path2 = os.path.join(image_dir, fname2)
            kps1, desc1 = image_features[idx1]
            kps2, desc2 = image_features[idx2]
            f1 = focal_lengths[idx1]
            f2 = focal_lengths[idx2]
            cam_poses, pts3D, metrics, all_matches, inlier_matches = two_view_triangulation(
                path1, path2,
                kps1, desc1,
                kps2, desc2,
                f1, f2,
                ratio_thresh=0.9
            )
            # Stocke les statistiques de cette paire
            pair_stats.append({
                "img1": fname1,
                "img2": fname2,
                "num_kp1": len(kps1),
                "num_kp2": len(kps2),
                "raw_matches": metrics["raw_matches"] if metrics else 0,
                "good_matches": metrics["good_matches"] if metrics else 0,
                "inliers_F": metrics["inliers_F"] if metrics else 0,
                "inliers_pose": metrics["inliers_pose"] if metrics else 0
            })
            # On peut aussi visualiser la paire si la triangulation a réussi
            if cam_poses is not None:
                print("Triangulation réussie pour cette paire.")
                visualize_matches(path1, path2, kps1, kps2, inlier_matches, window_name="Appariements Inliers")
                successful_pairs.append({
                    "img1": fname1,
                    "img2": fname2,
                    "cam_poses": cam_poses,
                    "pts3D": pts3D,
                    "metrics": metrics
                })
                # Ici, on ne casse pas la boucle pour collecter toutes les stats
    # Affichage des statistiques sous forme de courbes
    plot_stats(pair_stats)

    # Si au moins une paire a réussi, on écrit les fichiers de sortie pour la première paire réussie.
    if successful_pairs:
        best = successful_pairs[0]
        write_bundle_file("bundle.out", best["cam_poses"], best["pts3D"])
        write_ply_file("reconstruction.ply", best["cam_poses"], best["pts3D"])
        print("Reconstruction 3D terminée (2 vues).")
    else:
        print("Aucune paire n'a permis une triangulation réussie.")

# ------------------------------
# 9) Fonctions de plotting
# ------------------------------
def plot_stats(stats):
    # Au lieu d'utiliser les noms de fichiers comme label, on utilise un simple index
    x_indices = range(len(stats))
    
    good_vals = [d["good_matches"] for d in stats]
    inliers_pose_vals = [d["inliers_pose"] for d in stats]
    inliers_F_vals = [d["inliers_F"] for d in stats]

    # Plot 1 : Nombre de bons matches par paire
    plt.figure(figsize=(10, 5))
    plt.bar(x_indices, good_vals, color='steelblue')
    plt.xlabel("Index de la paire")
    plt.ylabel("Nombre de bons matches")
    plt.title("Bons matches par paire")
    plt.tight_layout()
    plt.show()

    # Plot 2 : Nombre d'inliers de recoverPose par paire
    plt.figure(figsize=(10, 5))
    plt.bar(x_indices, inliers_pose_vals, color='orange')
    plt.xlabel("Index de la paire")
    plt.ylabel("Inliers recoverPose")
    plt.title("Inliers recoverPose par paire")
    plt.tight_layout()
    plt.show()

    # Plot 3 : Scatter plot : bons matches vs inliers_F
    plt.figure(figsize=(7, 5))
    plt.scatter(good_vals, inliers_F_vals, color='green')
    plt.xlabel("Nombre de bons matches")
    plt.ylabel("Inliers Fundamental")
    plt.title("Corrélation : bons matches vs inliers F")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sfm_notredame.py <image_directory> [ref_notredame.jpg]")
        sys.exit(1)

    image_directory = sys.argv[1]
    reference_image = sys.argv[2] if len(sys.argv) > 2 else "notredame_reference.jpg"
    main(image_directory, reference_image)
