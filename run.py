import cv2
import numpy as np
from tqdm import tqdm
import argparse
from relpose.utils import read_video_np, visualize_T_w2c_rotations, visualize_rotation_angles
from pathlib import Path
from relpose.matcher_wrapper import Matcher
from relpose.solver_two_pairs import TwoPairSolver, CameraParams


def process_video_T_w2c_list(frames, matcher: Matcher, solver: TwoPairSolver):
    T_w2c_list = [np.eye(4)]  # cam poses are defined as T_w2c @ p_w = p_c
    prev_frame = frames[0]
    for frame_idx in tqdm(range(1, len(frames))):
        curr_frame = frames[frame_idx]

        # Match frames
        pts0, pts1 = matcher.match_np(prev_frame, curr_frame)
        T_delta = solver.solve(pts0, pts1)  # T_delta = T_curr @ T_last^-1

        # Compute current frame's transformation matrix
        T_w2c_list.append(T_delta @ T_w2c_list[-1])

        # Current frame becomes previous frame for next iteration
        prev_frame = curr_frame

    return T_w2c_list


def parse_args():
    parser = argparse.ArgumentParser(description="Camera T_w2c_list estimation from video")
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
    F, H, W, C = frames.shape
    print(f"Downsampled frames shape: {frames.shape}")

    matcher: Matcher = Matcher(args.method)
    solver: TwoPairSolver = TwoPairSolver(CameraParams(W, H))
    T_w2c_list = process_video_T_w2c_list(frames, matcher, solver)

    # Visualize the rotation T_w2c_list
    visualize_T_w2c_rotations(T_w2c_list, args.output_dir)

    # Also visualize additional rotation representations
    visualize_rotation_angles(T_w2c_list, args.output_dir)


if __name__ == "__main__":
    main()
