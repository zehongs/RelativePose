import cv2
import numpy as np
from dataclasses import dataclass
import pycolmap


@dataclass
class CameraParams:
    width: int
    height: int
    focal_length: float = None  # Use sqrt(width^2 + height^2) if not provided FOV~=53°
    cx: float = None  # Use half of width if not provided
    cy: float = None  # Use half of height if not provided


class Cv2RansacEssentialSolver:
    def __init__(self, camera_params: CameraParams):
        width = camera_params.width
        height = camera_params.height
        focal_length = camera_params.focal_length
        if focal_length is None:
            focal_length = (width**2 + height**2) ** 0.5
        cx = camera_params.cx
        cy = camera_params.cy
        if cx is None:
            cx = width / 2
        if cy is None:
            cy = height / 2

        self.camera_matrix = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])

    def get_K(self):
        """
        Returns:
            K: np.ndarray, shape (3, 3), dtype=np.float32
        """
        return self.camera_matrix

    def solve(self, pts0, pts1):
        # Find essential matrix with stricter RANSAC
        E, mask = cv2.findEssentialMat(
            pts0,
            pts1,
            self.camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts0, pts1, self.camera_matrix, mask=mask)

        return R, t


class PycolmapRansacEssentialSolver:
    def __init__(self, camera_params: CameraParams):
        width = camera_params.width
        height = camera_params.height
        focal_length = camera_params.focal_length
        if focal_length is None:
            focal_length = (width**2 + height**2) ** 0.5
        cx = camera_params.cx
        cy = camera_params.cy
        if cx is None:
            cx = width / 2
        if cy is None:
            cy = height / 2
        self.camera_matrix = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])

        # Set up pycolmap
        self.camera = pycolmap.Camera(
            camera_id=0,
            model="SIMPLE_PINHOLE",
            width=width,
            height=height,
            params=[focal_length, cx, cy],
        )

        # Configure options for consecutive frames
        self.options = pycolmap.TwoViewGeometryOptions(
            min_num_inliers=10,
            min_E_F_inlier_ratio=0.8,
            max_H_inlier_ratio=0.9,
            compute_relative_pose=True,
        )
        print(self.options.summary())

    def get_K(self):
        return self.camera_matrix

    def solve(self, pts0, pts1):
        matches = np.stack([np.arange(len(pts0)), np.arange(len(pts0))], axis=-1)
        answer = pycolmap.estimate_calibrated_two_view_geometry(
            self.camera,
            pts0.astype(np.float64),
            self.camera,
            pts1.astype(np.float64),
            matches=matches,
            options=self.options,
        )

        # cam2_from_cam1 means T_0_to_1 in our language
        Rt = answer.cam2_from_cam1.matrix().astype(np.float32)  # shape (3, 4)
        T = np.eye(4)
        T[:3] = Rt
        return T


two_pair_solver_map = {
    # "cv2": Cv2RansacEssentialSolver,  # This is not stable
    "pycolmap": PycolmapRansacEssentialSolver,
}


class TwoPairSolver:
    def __init__(self, params: CameraParams, solver: str = "pycolmap"):
        self.solver = two_pair_solver_map[solver](params)

    def get_K(self):
        """
        Returns:
            K: np.ndarray, shape (3, 3), dtype=np.float32
        """
        return self.solver.get_K()

    def solve(self, pts0, pts1):
        """
        Args:
            pts0: np.ndarray, shape (N, 2), dtype=np.float32
            pts1: np.ndarray, shape (N, 2), dtype=np.float32
        Returns:
            T: np.ndarray, shape (4, 4), dtype=np.float32
        """
        return self.solver.solve(pts0, pts1)
