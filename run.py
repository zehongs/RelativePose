import cv2
import numpy as np
from tqdm import tqdm
import argparse
from relpose.utils import read_video_np, visualize_rotation_trajectory, visualize_rotation_angles
from pathlib import Path


def compute_relative_pose(img0, img1, use_sift=True):
    """Compute relative pose between two images using SIFT features."""
    # Convert images to grayscale
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    if use_sift:
        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Find keypoints and descriptors
        kp0, des0 = sift.detectAndCompute(gray0, None)
        kp1, des1 = sift.detectAndCompute(gray1, None)

        # Match descriptors using FLANN matcher (better for SIFT)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des0, des1, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    else:
        # Original ORB method
        orb = cv2.ORB_create(nfeatures=2000)
        kp0, des0 = orb.detectAndCompute(gray0, None)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des0, des1)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:100]  # Take top 100 matches

    # Ensure we have enough matches
    if len(good_matches) < 8:
        print(
            f"Warning: Only {len(good_matches)} matches found, which might not be enough for reliable pose estimation"
        )
        # Pad with more matches if available
        if not use_sift and len(matches) > len(good_matches):
            good_matches = matches[: min(100, len(matches))]

    # Extract matched point coordinates
    pts0 = np.float32([kp0[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute essential matrix and recover pose
    # Use approximate camera matrix
    height, width = gray0.shape
    focal_length = 0.8 * width  # Approximation
    camera_matrix = np.array([[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]])

    # Find essential matrix with stricter RANSAC
    E, mask = cv2.findEssentialMat(pts0, pts1, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Recover pose
    _, R, t, mask = cv2.recoverPose(E, pts0, pts1, camera_matrix, mask=mask)

    # Create transformation matrix T_0to1
    T_0to1 = np.eye(4)
    T_0to1[:3, :3] = R
    T_0to1[:3, 3] = t.reshape(3)

    return T_0to1, good_matches, kp0, kp1


def process_video_trajectory(frames):
    """Process entire video to compute the camera trajectory."""
    F, H, W, C = frames.shape
    print(f"Frames shape: {frames.shape}")

    # Read the first frame
    prev_frame = frames[0]

    # Initialize trajectory with identity matrix for frame 0
    trajectory = [np.eye(4)]  # T_0_to_0 is identity
    relative_transforms = []

    # Process frame pairs
    for frame_idx in tqdm(range(1, F)):
        curr_frame = frames[frame_idx]

        # Compute relative pose between consecutive frames
        T_prev_to_curr, matches, kp_prev, kp_curr = compute_relative_pose(prev_frame, curr_frame)
        relative_transforms.append(T_prev_to_curr)

        # Compute the global transformation from frame 0 to current frame
        # T_0_to_t = T_(t-1)_to_t @ T_0_to_(t-1)
        T_0_to_curr = trajectory[-1] @ T_prev_to_curr

        trajectory.append(T_0_to_curr)

        # Current frame becomes previous frame for next iteration
        prev_frame = curr_frame

    return trajectory, relative_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Camera trajectory estimation from video")
    parser.add_argument("--video", type=str, default="inputs/0307.mp4", help="Path to input video file")
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        choices=["orb", "sift"],
        default="orb",
        help="Feature detection algorithm to use (default: orb)",
    )
    parser.add_argument("--step", type=int, default=8, help="Process every Nth frame (default: 8)")
    parser.add_argument("-ds", "--downsample", type=float, default=0.5, help="Downsample factor (default: 1)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory (default: outputs)")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # # Option 1: Original functionality - compute relative pose between frames 0 and 10
    # frame_indices = [0, 10]
    # frames = extract_frames(video_path, frame_indices)
    # img0, img10 = frames
    # T_0to10, matches, kp0, kp10 = compute_relative_pose(img0, img10)
    # print("Direct Transformation Matrix T_0to10:")
    # print(T_0to10)
    # visualize_matches(img0, img10, kp0, kp10, matches)

    # Option 2: Compute full trajectory
    print("\nComputing full camera trajectory...")
    frames = read_video_np(args.video, scale=args.downsample)
    frames = frames[:: args.step]
    trajectory, relative_transforms = process_video_trajectory(frames)

    # Visualize the rotation trajectory
    visualize_rotation_trajectory(trajectory, args.output_dir)

    # Also visualize additional rotation representations
    visualize_rotation_angles(trajectory, args.output_dir)


if __name__ == "__main__":
    main()
