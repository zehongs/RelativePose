import cv2
import numpy as np
from tqdm import tqdm
import argparse
from relpose.utils import read_video_frame_np, visualize_matches
from pathlib import Path
from relpose.wrapper import Matcher


def parse_args():
    parser = argparse.ArgumentParser(description="Draw matches between two frames")
    parser.add_argument("--video", type=str, default="inputs/0307.mp4", help="Path to input video file")
    parser.add_argument("--index0", type=int, default=0, help="Index of the first frame")
    parser.add_argument("--index1", type=int, default=10, help="Index of the second frame")
    parser.add_argument("-m", "--method", type=str, default="sift")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory (default: outputs)")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Video path: {args.video}, index0: {args.index0}, index1: {args.index1}")

    img0 = read_video_frame_np(args.video, args.index0)
    img1 = read_video_frame_np(args.video, args.index1)
    print(f"img0 shape: {img0.shape}, img1 shape: {img1.shape}")

    matcher = Matcher(args.method)
    pts0, pts1 = matcher.match_np(img0, img1)
    print(f"pts0 shape: {pts0.shape}, pts1 shape: {pts1.shape}")

    # 可视化匹配结果
    visualize_matches(img0, img1, pts0, pts1, args.output_dir)


if __name__ == "__main__":
    main()
