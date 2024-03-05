from __future__ import print_function

import urllib
import bz2
import os
import numpy as np
import json
import bz2
import os
import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import time
from scipy.optimize import least_squares
import cv2
from pathlib import Path
from enum import Enum

N_PARAMS = 15
CAMERA_IDX_TO_NAME = {}


class AdjustmentTypes(Enum):
    ALL = "all"
    EXTRINSIC_ONLY = "extrinsic"
    EXTRINSIC_AND_POINTS_ONLY = "extrinsic_and_points"


ADJUSTMENT_TYPE = AdjustmentTypes.EXTRINSIC_AND_POINTS_ONLY


def build_calibrations(intrinsic, extrinsic):
    calibration = {}
    intrinsic = intrinsic.replace("cam0", "cam")
    with open(intrinsic, "r") as icf:
        data = json.load(icf)
        calibration["cam_matrix"] = np.array(data["intrinsic"])
        calibration["distortion_coefficients"] = np.array(data["distortion_coefficients"]).reshape((len(data["distortion_coefficients"]), 1))
    with open(extrinsic, "r") as ecf:
        data = json.load(ecf)
        calibration["tvec"] = np.array(data["tvec"]).reshape((len(data["tvec"]), 1))
        calibration["rvec"] = np.array(data["rvec"]).reshape((len(data["rvec"]), 1))
    return calibration


def read_calibration(cameras, intrinsic, extrinsic):
    calibrations = {}
    for camera in cameras:
        intrinsic_path = intrinsic[0] + camera + intrinsic[1]
        extrinsic_path = extrinsic[0] + camera + extrinsic[1]
        calibrations[camera] = build_calibrations(intrinsic_path, extrinsic_path)
    return calibrations


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid="ignore"):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def project(points, camera_params, camera_indices, show_projections: bool = False):
    """Convert 3-D points to 2-D by projecting onto images."""
    # points_proj = rotate(points, camera_params[:, :3])
    # points_proj += camera_params[:, 3:6]
    # points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]

    rvec = camera_params[:, :3]
    tvec = camera_params[:, 3:6]
    fx = camera_params[:, 6]
    fy = camera_params[:, 7]
    cx = camera_params[:, 8]
    cy = camera_params[:, 9]
    dist_coeffs = camera_params[:, 10:]  # 5 distortion coefficients

    cam_matrix = np.eye(3)
    points_proj = np.empty((len(camera_params), 2))
    for i in range(len(camera_params)):
        cam_matrix[0, 0] = fx[i]
        cam_matrix[1, 1] = fy[i]
        cam_matrix[0, 2] = cx[i]
        cam_matrix[1, 2] = cy[i]
        points_proj[i], _ = cv2.projectPoints(points[i], rvec=rvec[i], tvec=tvec[i], cameraMatrix=cam_matrix, distCoeffs=dist_coeffs[i])

        if show_projections:
            # Show the 2D projections of the 3D points on the images
            wname = "test"
            cv2.namedWindow(wname)
            image = cv2.imread(f"/home/era/code/NEON/camera-calibration-extrinsic/{CAMERA_IDX_TO_NAME[camera_indices[i]]}/frame_000.jpg")
            frame = np.copy(image)
            cv2.circle(frame, points_proj[i].astype(np.int16), 3, (0, 0, 255), 2)
            cv2.imshow(wname, frame)
            cv2.waitKey(0)

    # n = np.sum(points_proj**2, axis=1)
    # r = 1 + k1 * n + k2 * n**2
    # points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def residuals_all(params: np.ndarray, n_cameras: int, n_points: int, camera_indices: np.ndarray, point_indices: np.ndarray, points_2d: np.ndarray):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    global N_PARAMS
    camera_params = params[: n_cameras * N_PARAMS].reshape((n_cameras, N_PARAMS))
    points_3d = params[n_cameras * N_PARAMS :].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], camera_indices)
    return (points_proj - points_2d).ravel()


def residuals_extrinsic_only(
    params: np.ndarray,
    n_cameras: int,
    n_points: int,
    camera_indices: np.ndarray,
    point_indices: np.ndarray,
    points_2d: np.ndarray,
    points_3d: np.ndarray,
    camera_extrinsic_params: np.ndarray,
):
    global N_PARAMS
    camera_params = params[: n_cameras * N_PARAMS].reshape((n_cameras, N_PARAMS))
    camera_params = np.concatenate([camera_params, camera_extrinsic_params], axis=1)
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], camera_indices)
    return (points_proj - points_2d).ravel()


def residuals_extrinsic_and_points(
    params: np.ndarray,
    n_cameras: int,
    n_points: int,
    camera_indices: np.ndarray,
    point_indices: np.ndarray,
    points_2d: np.ndarray,
    camera_extrinsic_params: np.ndarray,
):
    global N_PARAMS
    camera_params = params[: n_cameras * N_PARAMS].reshape((n_cameras, N_PARAMS))
    points_3d = params[n_cameras * N_PARAMS :].reshape((n_points, 3))
    camera_params = np.concatenate([camera_params, camera_extrinsic_params], axis=1)
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], camera_indices)
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    global N_PARAMS
    m = camera_indices.size * 2
    n = n_cameras * N_PARAMS + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(N_PARAMS):
        A[2 * i, camera_indices * N_PARAMS + s] = 1
        A[2 * i + 1, camera_indices * N_PARAMS + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * N_PARAMS + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * N_PARAMS + point_indices * 3 + s] = 1

    return A


def calibrations_to_params(calibrations: dict) -> np.ndarray:
    camera_params = np.empty((len(calibrations), 15))
    # rot, trans, focal and distortion params
    for i, data in enumerate(calibrations.values()):
        # The intrinsic matrix is in the form:
        # f_x  0    c_x
        # 0    f_y  c_y
        # 0    0    1
        f_x = data["cam_matrix"][0, 0]
        f_y = data["cam_matrix"][1, 1]
        c_x = data["cam_matrix"][0, 2]
        c_y = data["cam_matrix"][1, 2]
        intrinsics_data = np.array([f_x, f_y, c_x, c_y])

        camera_params[i] = np.concatenate(
            (data["rvec"].squeeze(1), data["tvec"].squeeze(1), intrinsics_data, data["distortion_coefficients"].squeeze(1))
        )

    return camera_params


def read_points(cameras, points_path_pattern):
    points_2d, points_3d_dict, point_indices, camera_indices = [], dict(), [], []
    for cam_num, camera in enumerate(cameras):
        CAMERA_IDX_TO_NAME[cam_num] = camera

        points_path = points_path_pattern[0] + camera + points_path_pattern[1]
        with open(points_path, "r") as f:
            data = json.load(f)
            points_2d_i = np.array(data["image_points"])
            points_3d_i = np.array(data["object_points"])

        points_2d.append(points_2d_i)
        camera_indices.extend([cam_num] * len(points_2d_i))

        for point_3d_i in points_3d_i:
            point_3d_i_str = str(point_3d_i)
            if point_3d_i_str not in points_3d_dict:
                points_3d_dict[point_3d_i_str] = [len(points_3d_dict), point_3d_i]

            point_indices.append(points_3d_dict[point_3d_i_str][0])

    points_3d = np.empty((len(points_3d_dict), 3))
    for i, points_3d_i in enumerate(points_3d_dict.values()):
        points_3d[i] = points_3d_i[1]

    points_2d = np.concatenate(points_2d)
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)

    return points_2d, points_3d, point_indices, camera_indices


def store_bundle_adjusted_params(camera_params: np.ndarray, intrinsic_path_pattern, extrinsic_path_pattern):
    for i, camera_param in enumerate(camera_params):
        camera = f"{CAMERA_IDX_TO_NAME[i]}"

        if ADJUSTMENT_TYPE == AdjustmentTypes.ALL:
            intrinsic_path = intrinsic_path_pattern[0] + camera + intrinsic_path_pattern[1]
            Path(intrinsic_path).parent.mkdir(parents=True, exist_ok=True)
            with open(intrinsic_path, "w") as f:
                cam_matrix = np.eye(3)
                cam_matrix[0, 0] = camera_param[6]
                cam_matrix[1, 1] = camera_param[7]
                cam_matrix[0, 2] = camera_param[8]
                cam_matrix[1, 2] = camera_param[9]
                data = {"intrinsic": cam_matrix.tolist(), "distortion_coefficients": camera_param[10:].tolist()}
                json.dump(data, f)

        extrinsic_path = extrinsic_path_pattern[0] + camera + extrinsic_path_pattern[1]
        Path(extrinsic_path).parent.mkdir(parents=True, exist_ok=True)
        with open(extrinsic_path, "w") as f:
            data = {}
            data["rvec"] = camera_param[:3].tolist()
            data["tvec"] = camera_param[3:6].tolist()
            json.dump(data, f)


def main():
    global N_PARAMS
    cameras = ["cam01", "cam02", "cam03", "cam04", "cam05", "cam06", "cam07", "cam08", "cam09", "cam10", "cam11", "cam12", "cam13"]
    intrinsic_path_pattern = ["/mnt/staff-bulk/ewi/insy/SPCDataSets/conflab-mm/processed/camera-calibration/", "/intrinsic.json"]
    extrinsic_path_pattern = ["/home/era/code/NEON/camera-calibration-extrinsic/", "/extrinsic.json"]
    points_path_pattern = ["/home/era/code/NEON/camera-calibration-extrinsic/", "/calibration_points.json"]
    calibrations = read_calibration(cameras, intrinsic_path_pattern, extrinsic_path_pattern)
    camera_params = calibrations_to_params(calibrations)
    points_2d, points_3d, point_indices, camera_indices = read_points(cameras, points_path_pattern)

    n_cameras = len(cameras)
    n_points = len(points_3d)

    if ADJUSTMENT_TYPE == AdjustmentTypes.ALL:
        N_PARAMS = 15
        x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

        f0 = residuals_all(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
        fixed_args = (n_cameras, n_points, camera_indices, point_indices, points_2d)
        function_to_optimize = residuals_all
    elif ADJUSTMENT_TYPE == AdjustmentTypes.EXTRINSIC_ONLY:
        N_PARAMS = 6
        x0 = camera_params[:, :6].ravel()  # only the extrinsic parameters

        f0 = residuals_extrinsic_only(x0, n_cameras, n_points, camera_indices, point_indices, points_2d, points_3d, camera_params[:, 6:])
        fixed_args = (n_cameras, n_points, camera_indices, point_indices, points_2d, points_3d, camera_params[:, 6:])
        function_to_optimize = residuals_extrinsic_only
    elif ADJUSTMENT_TYPE == AdjustmentTypes.EXTRINSIC_AND_POINTS_ONLY:
        N_PARAMS = 6
        x0 = np.hstack((camera_params[:, :6].ravel(), points_3d.ravel()))

        f0 = residuals_extrinsic_and_points(x0, n_cameras, n_points, camera_indices, point_indices, points_2d, camera_params[:, 6:])
        fixed_args = (n_cameras, n_points, camera_indices, point_indices, points_2d, camera_params[:, 6:])
        function_to_optimize = residuals_extrinsic_and_points

    plt.plot(f0)

    # A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    t0 = time.time()
    res = least_squares(
        function_to_optimize,
        x0,
        # jac_sparsity=A,
        verbose=2,
        x_scale="jac",
        ftol=1e-4,
        method="trf",
        # method="lm",
        args=fixed_args,
    )
    t1 = time.time()

    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    plt.plot(res.fun)

    new_camera_params = res.x[: n_cameras * N_PARAMS].reshape((n_cameras, N_PARAMS))

    if ADJUSTMENT_TYPE == AdjustmentTypes.ALL:
        new_points_3d = res.x[n_cameras * N_PARAMS :].reshape((n_points, 3))

        new_extrinsic_path_pattern = ["/home/era/code/NEON/camera-calibration-bundle-adjusted/", "/extrinsic.json"]
        new_intrinsic_path_pattern = ["/home/era/code/NEON/camera-calibration-bundle-adjusted/", "/intrinsic.json"]
    elif ADJUSTMENT_TYPE == AdjustmentTypes.EXTRINSIC_ONLY:
        new_extrinsic_path_pattern = ["/home/era/code/NEON/camera-calibration-bundle-adjusted-extrinsic-only/", "/extrinsic.json"]
        new_intrinsic_path_pattern = None
    elif ADJUSTMENT_TYPE == AdjustmentTypes.EXTRINSIC_AND_POINTS_ONLY:
        new_extrinsic_path_pattern = ["/home/era/code/NEON/camera-calibration-bundle-adjusted-extrinsic-and-points/", "/extrinsic.json"]
        new_intrinsic_path_pattern = None
    store_bundle_adjusted_params(new_camera_params, new_intrinsic_path_pattern, new_extrinsic_path_pattern)


if __name__ == "__main__":
    main()
