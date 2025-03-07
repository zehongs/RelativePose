import cv2
import numpy as np
from tqdm import tqdm
import argparse
from relpose.utils import read_video_np, visualize_rotation_trajectory, visualize_rotation_angles
from pathlib import Path
from relpose.wrapper import Matcher


def compute_relative_pose(pts0, pts1, H, W):
    """Compute relative pose between two images using SIFT features."""
    # Compute essential matrix and recover pose
    # Use approximate camera matrix
    focal_length = 0.8 * W  # Approximation
    camera_matrix = np.array([[focal_length, 0, W / 2], [0, focal_length, H / 2], [0, 0, 1]])

    # Find essential matrix with stricter RANSAC
    E, mask = cv2.findEssentialMat(pts0, pts1, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Recover pose
    _, R, t, mask = cv2.recoverPose(E, pts0, pts1, camera_matrix, mask=mask)

    # Create transformation matrix T_0to1
    T_0to1 = np.eye(4)
    T_0to1[:3, :3] = R
    T_0to1[:3, 3] = t.reshape(3)

    return T_0to1


def process_video_trajectory(frames, matcher: Matcher):
    """Process entire video to compute the camera trajectory."""
    F, H, W, C = frames.shape

    # Initialize trajectory with identity matrix for frame 0
    trajectory = [np.eye(4)]  # T_0_to_0 is identity
    relative_transforms = []

    # Process frame pairs
    prev_frame = frames[0]
    for frame_idx in tqdm(range(1, len(frames))):
        curr_frame = frames[frame_idx]

        # Match frames
        pts0, pts1 = matcher.match_np(prev_frame, curr_frame)

        # Compute relative pose between consecutive frames
        T_prev_to_curr = compute_relative_pose(pts0, pts1, H, W)
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
    parser.add_argument("-m", "--method", type=str, default="sift")
    parser.add_argument("--step", type=int, default=8, help="Process every Nth frame (default: 8)")
    parser.add_argument("-ds", "--downsample", type=float, default=0.5, help="Downsample factor (default: 1)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory (default: outputs)")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Video path: {args.video}")

    frames = read_video_np(args.video, scale=args.downsample)
    frames = frames[:: args.step]
    print(f"Downsampled frames shape: {frames.shape}")

    matcher: Matcher = Matcher(args.method)
    trajectory, relative_transforms = process_video_trajectory(frames, matcher)

    # Visualize the rotation trajectory
    visualize_rotation_trajectory(trajectory, args.output_dir)

    # Also visualize additional rotation representations
    visualize_rotation_angles(trajectory, args.output_dir)


if __name__ == "__main__":
    main()
