# Relative Pose

1. `run_pair.py`: Match image0 and image1, draw matches.
2. `run.py`: Compute camera poses of a given video. Camera poses are defined as T_i @ p_world = p_cam.

## Method

1. OpenCV SIFT and ORB


## Installation

```bash
# Match and solve camera poses
pip install opencv-python pycolmap

# Read videos
pip install video-reader-rs  # read videos
apt install ffmpeg   # ffmpeg is required
```

