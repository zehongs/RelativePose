import argparse
from relpose.utils import visualize_T_w2c_rotations, visualize_rotation_angles
from pathlib import Path
from relpose.simple_vo import SimpleVO


def parse_args():
    parser = argparse.ArgumentParser(description="Camera T_w2c_list estimation from video")
    parser.add_argument("--video", type=str, default="assets/emdb_example.mp4", help="Path to input video file")
    parser.add_argument("-m", "--method", type=str, default="sift")
    parser.add_argument("--step", type=int, default=8, help="Process every Nth frame (default: 8)")
    parser.add_argument("-ds", "--downsample", type=float, default=0.5, help="Downsample factor (default: 1)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory (default: outputs)")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Video path: {args.video}")

    # Run VO
    vo = SimpleVO(args.video, scale=args.downsample, step=args.step, method=args.method, f_mm=24)
    T_w2c_list = vo.compute()

    # Visualize the rotation T_w2c_list
    visualize_T_w2c_rotations(T_w2c_list, args.output_dir)

    # Also visualize additional rotation representations
    visualize_rotation_angles(T_w2c_list, args.output_dir)


if __name__ == "__main__":
    main()
