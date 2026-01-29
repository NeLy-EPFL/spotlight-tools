import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import cv2
import tyro
from pathlib import Path
from tqdm import tqdm

import spotlight_tools.file_format_versions as versions
from spotlight_tools.calibration.charuco import (
    ChArUcoBoard,
    detect_charuco,
    plot_charuco_detections,
)
from spotlight_tools.calibration.model import (
    ransac_filter_outliers,
    visualize_ransac_results,
)


def open_jpg_image(path: str) -> np.ndarray:
    image = cv2.imread(str(path))
    assert len(image.shape) == 3, f"Image {path} is not a 3-channel image."
    return image[:, :, 0]


def open_tif_image_and_normalize(path: str) -> np.ndarray:
    """Read 16-bit tif image and normalize to 0-255 range based on the 5th
    and 95th percentiles."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    q20 = np.percentile(image, 20)
    q80 = np.percentile(image, 80)
    image = (image - q20) / (q80 - q20)
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    return image


def gather_calibration_points(
    board: ChArUcoBoard,
    calibration_image_dir: Path,
    camera: str,
    charuco_detection_viz_dir: Path | None = None,
) -> pd.DataFrame:
    if not camera in ("behavior_camera", "muscle_camera"):
        raise ValueError(f"camera must be either 'behavior_camera' or 'muscle_camera'")

    all_data = []

    suffix = "tif" if camera == "muscle_camera" else "jpg"
    files = sorted(list(calibration_image_dir.glob(f"*.{suffix}")))
    for path in tqdm(files, disable=None, desc="Detecting Charuco markers"):
        _parts = path.stem.split("_")
        physical_x_mm = float(_parts[2].replace("x", ""))
        physical_y_mm = float(_parts[3].replace("y", ""))
        if camera == "muscle_camera":
            image = open_tif_image_and_normalize(str(path))
        elif camera == "behavior_camera":
            image = open_jpg_image(str(path))
        ids, corners = detect_charuco(image, board, camera, horizontal_flip=False)
        if charuco_detection_viz_dir is not None:
            fig, ax = plt.subplots()
            plot_charuco_detections(fig, ax, image, ids, corners)
            fig.savefig(charuco_detection_viz_dir / f"{path.stem}.png")
            plt.close(fig)
        if ids is None:
            continue
        for i, corner_id in enumerate(ids):
            phys_x, phys_y = board.corner_id_to_xy_mm(corner_id)
            data = {
                "corner_id": corner_id,
                "stage_x_mm": physical_x_mm,
                "stage_y_mm": physical_y_mm,
                "pixel_x_px": corners[i][0][0],
                "pixel_y_px": corners[i][0][1],
                "physical_x_mm": phys_x,
                "physical_y_mm": phys_y,
                "image_path": str(path),
            }
            all_data.append(data)

    return pd.DataFrame(all_data)


def A_to_B(A):
    M = A[:, 2:4]  # pixel x and y weights
    M_inv = np.linalg.inv(M)
    A_stage = A[:, 0:2]  # stage position
    A_bias = A[:, 4:5]  # bias
    B_stage = -M_inv @ A_stage
    B_physical = M_inv
    B_bias = -M_inv @ A_bias
    B = np.hstack([B_stage, B_physical, B_bias])
    return B


def fit_calibration_model(
    profile_dir: str = "~/Spotlight/default/",
    muscle_camera: bool = False,
    arena_width: float = 48,
    arena_height: float = 72,
    visualize_charuco_detections: bool = False,
) -> None:
    """
    Fit the calibration model for the Charuco board.

    Args:
        profile_dir (str):
            Path to the profile directory.
        muscle_camera (bool):
            Whether to use fit calibration model for muscle camera.
        arena_width (float):
            Width of the arena in mm.
        arena_height (float):
            Height of the arena in mm.
        visualize_charuco_detections (bool):
            Whether to visualize the Charuco detections.
    """
    print("Fitting calibration model for behavior camera...")
    fit_calibration_model_one_camera(
        profile_dir=profile_dir,
        camera="behavior_camera",
        arena_width=arena_width,
        arena_height=arena_height,
        visualize_charuco_detections=visualize_charuco_detections,
    )

    if muscle_camera:
        print("Fitting calibration model for muscle camera...")
        fit_calibration_model_one_camera(
            profile_dir=profile_dir,
            camera="muscle_camera",
            arena_width=arena_width,
            arena_height=arena_height,
            visualize_charuco_detections=visualize_charuco_detections,
        )


def fit_calibration_model_one_camera(
    profile_dir: str = "~/Spotlight/default/",
    camera: str = "behavior_camera",
    arena_width: float = 48,
    arena_height: float = 72,
    visualize_charuco_detections: bool = False,
) -> None:
    # Expand the profile directory
    profile_dir = Path(profile_dir).expanduser()

    # Check if the calibration images directory exists and is not empty
    calibration_image_dir = Path(profile_dir) / f"calibration/aruco_scan/{camera}"
    if not calibration_image_dir.exists():
        raise FileNotFoundError(
            f"Directory {calibration_image_dir} does not exist. "
            "Run calibration scan first."
        )
    suffix = "jpg" if camera == "behavior_camera" else "tif"
    if not list(calibration_image_dir.glob(f"*.{suffix}")):
        raise FileNotFoundError(
            f"Directory {calibration_image_dir} does not contain any file with suffix "
            f".{suffix}. Run calibration scan first and make sure the data are saved "
            f"in the .{suffix} format."
        )

    # Create the directory for visualizing ChArUco detections if necessary
    charuco_detection_viz_dir = (
        Path(profile_dir) / f"calibration/charuco_scan_detection/{camera}"
    )
    if visualize_charuco_detections:
        charuco_detection_viz_dir.mkdir(exist_ok=True, parents=True)

    # Initialize the Charuco board
    charuco_board = ChArUcoBoard(
        arena_dim_mm=(arena_width, arena_height),
    )

    # Gather calibration points
    coordinates_df = gather_calibration_points(
        charuco_board,
        calibration_image_dir,
        camera,
        charuco_detection_viz_dir=(
            charuco_detection_viz_dir if visualize_charuco_detections else None
        ),
    )
    coordinates_df_path = (
        Path(profile_dir) / f"calibration/model/{camera}/calibration_points.csv"
    )
    coordinates_df_path.parent.mkdir(parents=True, exist_ok=True)
    coordinates_df.to_csv(coordinates_df_path, index=False)

    # Fit model
    ransac_result = ransac_filter_outliers(coordinates_df)
    visualize_ransac_results(coordinates_df, ransac_result)

    x_model = ransac_result["models"]["x"]
    y_model = ransac_result["models"]["y"]

    mat_pixel_to_physical = np.array(
        [
            [*x_model.coef_.squeeze(), x_model.intercept_],
            [*y_model.coef_.squeeze(), y_model.intercept_],
        ]
    )

    mat_physical_to_pixel = A_to_B(mat_pixel_to_physical)

    calibration_results = {
        "metadata": {
            # Assign a version number to the format of this calibration file
            # Follow semver for backward compatibility detection
            "file_format_version": {
                "major": versions.calibration_result_major,
                "minor": versions.calibration_result_minor,
                "patch": versions.calibration_result_patch,
            }
        },
        "stage_and_pixel_to_physical": {
            "physical_pos_x": {
                "stage_pos_x": float(mat_pixel_to_physical[0, 0]),
                "stage_pos_y": float(mat_pixel_to_physical[0, 1]),
                "pixel_pos_x": float(mat_pixel_to_physical[0, 2]),
                "pixel_pos_y": float(mat_pixel_to_physical[0, 3]),
                "bias": float(mat_pixel_to_physical[0, 4]),
            },
            "physical_pos_y": {
                "stage_pos_x": float(mat_pixel_to_physical[1, 0]),
                "stage_pos_y": float(mat_pixel_to_physical[1, 1]),
                "pixel_pos_x": float(mat_pixel_to_physical[1, 2]),
                "pixel_pos_y": float(mat_pixel_to_physical[1, 3]),
                "bias": float(mat_pixel_to_physical[1, 4]),
            },
        },
        "stage_and_physical_to_pixel": {
            "pixel_pos_x": {
                "stage_pos_x": float(mat_physical_to_pixel[0, 0]),
                "stage_pos_y": float(mat_physical_to_pixel[0, 1]),
                "physical_pos_x": float(mat_physical_to_pixel[0, 2]),
                "physical_pos_y": float(mat_physical_to_pixel[0, 3]),
                "bias": float(mat_physical_to_pixel[0, 4]),
            },
            "pixel_pos_y": {
                "stage_pos_x": float(mat_physical_to_pixel[1, 0]),
                "stage_pos_y": float(mat_physical_to_pixel[1, 1]),
                "physical_pos_x": float(mat_physical_to_pixel[1, 2]),
                "physical_pos_y": float(mat_physical_to_pixel[1, 3]),
                "bias": float(mat_physical_to_pixel[1, 4]),
            },
        },
    }

    calibration_results_path = (
        Path(profile_dir) / f"calibration/model/{camera}/calibration_result.yaml"
    )
    with open(calibration_results_path, "w") as f:
        yaml.dump(calibration_results, f)


def main():
    tyro.cli(fit_calibration_model)


if __name__ == "__main__":
    main()
