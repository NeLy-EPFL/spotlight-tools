import numpy as np
import yaml
import cv2
import tyro
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import spotlight_tools.file_format_versions as versions
from spotlight_tools.calibration.charuco import get_gizem_board
from spotlight_tools.scripts.fit_calibration import (
    open_jpg_image,
    open_tif_image_and_normalize,
)

def normalize_image(image: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalize image to range [0, 1]."""
    image = (image.astype(np.float32) - vmin) / (vmax - vmin)
    image = np.clip(image, 0, 1)
    return (image*255).astype(np.uint8)

def binarize_image_and_morph(image: np.ndarray, c: float, kernel: np.ndarray=[11, 11]) -> np.ndarray:
    """Binarize image using a threshold."""
    binary_image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,101,c)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    # binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return binary_image

def visualize_charuco_detection(
    image: np.ndarray,
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    marker_corners: np.ndarray,
    marker_ids: np.ndarray,
    camera: str,
) -> plt.Figure:
    """Visualize ChArUco detection results."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a copy for drawing (cv2 functions modify in-place)
    display_image = image.copy()
    
    # # Draw ArUco markers
    if marker_ids is not None and len(marker_ids) > 0:
        # Convert to 3-channel for colored drawings
        if len(display_image.shape) == 2:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2RGB)
        cv2.aruco.drawDetectedMarkers(display_image, marker_corners, marker_ids)
    
    # Display image
    if len(display_image.shape) == 2:
        ax.imshow(display_image, cmap="gray")
    else:
        ax.imshow(display_image)
    
    # Draw ChArUco corners
    if charuco_ids is not None and len(charuco_ids) > 0:
        for i, (corner, corner_id) in enumerate(zip(charuco_corners, charuco_ids)):
            x, y = corner.ravel()
            ax.plot(x, y, "ro", markersize=2)
            ax.text(x, y, str(corner_id[0]), color="yellow", fontsize=6)
    
    ax.set_title(f"{camera}: Detected {len(charuco_ids) if charuco_ids is not None else 0} ChArUco corners")
    ax.axis("off")
    
    return fig


def process_single_image_pair(
    behavior_path: Path,
    muscle_dir: Path,
    visualization_dir: Path | None = None,
) -> tuple:
    """
    Process a single image pair to detect and match ChArUco corners.
    
    Args:
        behavior_path: Path to behavior camera image
        muscle_dir: Directory containing muscle camera images
        visualization_dir: Optional directory to save detection visualizations
        
    Returns:
        Tuple of (stage_key, matching_data_dict) or (None, None) if processing failed
    """
    # Create CharucoBoard inside worker process (can't be pickled)
    charuco_board = get_gizem_board()
    
    # Parse stage position from filename
    parts = behavior_path.stem.split("_")
    stage_x = float(parts[2].replace("x", ""))
    stage_y = float(parts[3].replace("y", ""))
    stage_key = (stage_x, stage_y)
    
    # Find corresponding muscle camera image
    muscle_path = muscle_dir / behavior_path.name.replace(".jpg", ".tif")
    if not muscle_path.exists():
        print(f"Warning: No matching muscle image for {behavior_path.name}")
        return None, None
    
    # Load images
    behavior_image = open_jpg_image(str(behavior_path))
    # Min max normalize
    muscle_image = cv2.imread(str(muscle_path), cv2.IMREAD_UNCHANGED)
    muscle_image = np.clip(muscle_image, 0, 255).astype(np.uint8)

    behavior_image = binarize_image_and_morph(behavior_image, 0)
    muscle_image = binarize_image_and_morph(muscle_image, -5)

    # Detect ChArUco corners in both images
    beh_corners, beh_ids, beh_markers, beh_marker_ids = charuco_board.detect_corners(
        behavior_image, camera="behavior_camera", subpixel_refinement=True
    )
    muscle_corners, muscle_ids, muscle_markers, muscle_marker_ids = charuco_board.detect_corners(
        muscle_image, camera="muscle_camera", subpixel_refinement=True
    )
    
    # Visualize detections if requested
    if visualization_dir is not None:
        beh_viz_dir = visualization_dir / "behavior_camera"
        muscle_viz_dir = visualization_dir / "muscle_camera"
        beh_viz_dir.mkdir(parents=True, exist_ok=True)
        muscle_viz_dir.mkdir(parents=True, exist_ok=True)
        
        fig_beh = visualize_charuco_detection(
            behavior_image, beh_corners, beh_ids, beh_markers, beh_marker_ids, "behavior"
        )
        fig_beh.savefig(beh_viz_dir / f"{behavior_path.stem}.png", dpi=300, bbox_inches="tight")
        plt.close(fig_beh)
        
        fig_muscle = visualize_charuco_detection(
            muscle_image, muscle_corners, muscle_ids, muscle_markers, muscle_marker_ids, "muscle"
        )
        fig_muscle.savefig(muscle_viz_dir / f"{behavior_path.stem}.png", dpi=300, bbox_inches="tight")
        plt.close(fig_muscle)
    
    # Skip if either detection failed
    if beh_ids is None or muscle_ids is None:
        return None, None
    if len(beh_ids) == 0 or len(muscle_ids) == 0:
        return None, None
    
    # Find matching corner IDs
    beh_ids_flat = beh_ids.flatten()
    muscle_ids_flat = muscle_ids.flatten()
    common_ids = np.intersect1d(beh_ids_flat, muscle_ids_flat)
    
    if len(common_ids) < 4:
        print(f"Warning: Only {len(common_ids)} common corners found for stage position {stage_key}")
        return None, None
    
    # Extract matching corners
    matching_corners = []
    for corner_id in common_ids:
        beh_idx = np.where(beh_ids_flat == corner_id)[0][0]
        muscle_idx = np.where(muscle_ids_flat == corner_id)[0][0]
        
        beh_corner = beh_corners[beh_idx].ravel()
        muscle_corner = muscle_corners[muscle_idx].ravel()
        
        matching_corners.append((beh_corner, muscle_corner))
    
    matching_data_dict = {
        "behavior_corners": beh_corners,
        "behavior_ids": beh_ids,
        "muscle_corners": muscle_corners,
        "muscle_ids": muscle_ids,
        "matching_corners": matching_corners,
        "num_matches": len(matching_corners),
    }
    
    return stage_key, matching_data_dict


def gather_matching_corners(
    calibration_image_dir: Path,
    visualization_dir: Path | None = None,
    num_workers: int = -1,
) -> dict:
    """
    Gather matching ChArUco corners from behavior and muscle camera images.
    
    Args:
        calibration_image_dir: Directory containing calibration images
        visualization_dir: Optional directory to save detection visualizations
        num_workers: Number of parallel workers (-1 for all CPUs)
        
    Returns:
        Dictionary with stage positions as keys and matching corners as values
    """
    behavior_dir = calibration_image_dir / "behavior_camera"
    muscle_dir = calibration_image_dir / "muscle_camera"
    
    if not behavior_dir.exists():
        raise FileNotFoundError(
            f"Directory {behavior_dir} does not exist. "
            "Run calibration scan first."
        )
    if not muscle_dir.exists():
        raise FileNotFoundError(
            f"Directory {muscle_dir} does not exist. "
            "Run calibration scan first for muscle camera."
        )
    
    # Clean visualization directories if they exist (to avoid confusion with old visualizations)
    if visualization_dir is not None:
        beh_viz_dir = visualization_dir / "behavior_camera"
        muscle_viz_dir = visualization_dir / "muscle_camera"
        for viz_dir in [beh_viz_dir, muscle_viz_dir]:
            if viz_dir.exists():
                for file in viz_dir.iterdir():
                    if file.is_file():
                        file.unlink()
    
    # Get all image pairs
    behavior_files = sorted(list(behavior_dir.glob("*.jpg")))
    muscle_files = sorted(list(muscle_dir.glob("*.tif")))
    
    if len(behavior_files) == 0:
        raise FileNotFoundError(f"No .jpg files found in {behavior_dir}")
    if len(muscle_files) == 0:
        raise FileNotFoundError(f"No .tif files found in {muscle_dir}")
    
    # Process all image pairs in parallel
    print(f"Processing {len(behavior_files)} image pairs with {num_workers} workers...")
    results = Parallel(n_jobs=num_workers, backend="loky")(
        delayed(process_single_image_pair)(
            behavior_path, muscle_dir, visualization_dir
        )
        for behavior_path in tqdm(behavior_files, desc="Processing image pairs")
    )
    
    # Collect results into dictionary
    matching_data = {}
    for stage_key, data in results:
        if stage_key is not None and data is not None:
            matching_data[stage_key] = data
    
    return matching_data


def compute_homography_from_matches(matching_data: dict) -> tuple:
    """
    Compute homography matrix from matching corners across multiple images.
    
    Args:
        matching_data: Dictionary with matching corners data
        
    Returns:
        Tuple of (homography_matrix, inlier_mask, all_behavior_pts, all_muscle_pts)
    """
    # Collect all matching points
    all_behavior_pts = []
    all_muscle_pts = []
    
    for stage_key, data in matching_data.items():
        for beh_corner, muscle_corner in data["matching_corners"]:
            all_behavior_pts.append(beh_corner)
            all_muscle_pts.append(muscle_corner)
    
    all_behavior_pts = np.array(all_behavior_pts)
    all_muscle_pts = np.array(all_muscle_pts)
    
    print(f"Total matching corners collected: {len(all_behavior_pts)}")
    
    # Compute homography using RANSAC
    H_beh2muscle, inliers = cv2.findHomography(
        all_behavior_pts,
        all_muscle_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )
    
    num_inliers = np.sum(inliers)
    print(f"Inliers: {num_inliers}/{len(all_behavior_pts)} ({100*num_inliers/len(all_behavior_pts):.1f}%)")
    
    return H_beh2muscle, inliers, all_behavior_pts, all_muscle_pts


def visualize_homography_quality(
    H: np.ndarray,
    inliers: np.ndarray,
    behavior_pts: np.ndarray,
    muscle_pts: np.ndarray,
    output_path: Path,
    n_points_to_show: int = 500,
) -> None:
    """Visualize the quality of the homography fit."""
    # Transform behavior points using homography
    behavior_pts_homogeneous = np.hstack([behavior_pts, np.ones((len(behavior_pts), 1))])
    transformed_pts = (H @ behavior_pts_homogeneous.T).T
    transformed_pts = transformed_pts[:, :2] / transformed_pts[:, 2:3]
    
    # Compute reprojection errors
    errors = np.linalg.norm(transformed_pts - muscle_pts, axis=1)
    
    # Create visualization with 2 rows
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Scatter of errors
    ax = axes[0, 0]
    inlier_mask = inliers.flatten().astype(bool)
    ax.scatter(
        range(len(errors)),
        errors,
        c=["green" if inlier else "red" for inlier in inlier_mask],
        alpha=0.6,
        s=10,
    )
    ax.set_xlabel("Point index")
    ax.set_ylabel("Reprojection error (pixels)")
    ax.set_title("Reprojection Errors")
    ax.axhline(y=5.0, color="orange", linestyle="--", label="RANSAC threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Histogram of errors
    ax = axes[0, 1]
    ax.hist(errors[inlier_mask], bins=30, alpha=0.7, color="green", label="Inliers")
    ax.hist(errors[~inlier_mask], bins=30, alpha=0.7, color="red", label="Outliers")
    ax.set_xlabel("Reprojection error (pixels)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Error statistics
    ax = axes[0, 2]
    stats_text = f"""
    Total points: {len(errors)}
    Inliers: {np.sum(inlier_mask)} ({100*np.sum(inlier_mask)/len(errors):.1f}%)
    Outliers: {np.sum(~inlier_mask)} ({100*np.sum(~inlier_mask)/len(errors):.1f}%)
    
    Inlier errors:
      Mean: {np.mean(errors[inlier_mask]):.3f} px
      Median: {np.median(errors[inlier_mask]):.3f} px
      Std: {np.std(errors[inlier_mask]):.3f} px
      Max: {np.max(errors[inlier_mask]):.3f} px
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, family="monospace", verticalalignment="center")
    ax.axis("off")
    ax.set_title("Statistics")

    show_indices = np.random.choice(len(errors),
                                    size=min(n_points_to_show, len(errors)),
                                    replace=False)
    selected_inlier_mask = inlier_mask[show_indices]

    # Plot 4: Spatial distribution in behavior camera
    ax = axes[1, 0]
    scatter = ax.scatter(
        behavior_pts[show_indices, 0],
        behavior_pts[show_indices, 1],
        c=errors[show_indices],
        cmap="RdYlGn_r",
        s=4,
        alpha=0.7,
        vmin=0,
        vmax=10,
    )
    # Mark outliers with red circles
    if np.any(~selected_inlier_mask):
        ax.scatter(
            behavior_pts[show_indices][~selected_inlier_mask, 0],
            behavior_pts[show_indices][~selected_inlier_mask, 1],
            facecolors="none",
            edgecolors="red",
            s=10,
            linewidths=2,
            label="Outliers",
        )
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("Spatial Distribution (Behavior Camera)")
    ax.invert_yaxis()  # Image coordinates have y=0 at top
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Reprojection error (px)")
    if np.any(~inlier_mask):
        ax.legend()
    
    # Plot 5: Spatial distribution in muscle camera
    ax = axes[1, 1]
    scatter = ax.scatter(
        muscle_pts[show_indices, 0],
        muscle_pts[show_indices, 1],
        c=errors[show_indices],
        cmap="RdYlGn_r",
        s=4,
        alpha=0.7,
        vmin=0,
        vmax=10,
    )
    # Mark outliers with red circles
    if np.any(~selected_inlier_mask):
        ax.scatter(
            muscle_pts[show_indices][~selected_inlier_mask, 0],
            muscle_pts[show_indices][~selected_inlier_mask, 1],
            facecolors="none",
            edgecolors="red",
            s=10,
            linewidths=2,
            label="Outliers",
        )
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("Spatial Distribution (Muscle Camera)")
    ax.invert_yaxis()  # Image coordinates have y=0 at top
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Reprojection error (px)")
    if np.any(~inlier_mask):
        ax.legend()
    
    # Plot 6: Error vectors (showing direction of errors)
    ax = axes[1, 2]

    # Plot transformed behavior points and actual muscle points
    ax.scatter(
        muscle_pts[show_indices, 0],
        muscle_pts[show_indices, 1],
        c="blue",
        s=10,
        alpha=0.5,
        label="Actual (muscle)",
    )
    ax.scatter(
        transformed_pts[show_indices, 0],
        transformed_pts[show_indices, 1],
        c="orange",
        s=10,
        alpha=0.5,
        label="Transformed (behavior→muscle)",
    )
    
    # Draw error vectors for subset of points
    for idx in show_indices:
        if inlier_mask[idx]:
            color = "green"
            alpha = 0.3
        else:
            color = "red"
            alpha = 0.8
        ax.arrow(
            transformed_pts[idx, 0],
            transformed_pts[idx, 1],
            muscle_pts[idx, 0] - transformed_pts[idx, 0],
            muscle_pts[idx, 1] - transformed_pts[idx, 1],
            head_width=5,
            head_length=5,
            fc=color,
            ec=color,
            alpha=alpha,
            length_includes_head=True,
        )
    
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title(f"Error Vectors (showing {len(show_indices)} points)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Homography quality visualization saved to {output_path}")


def fit_homography(
    profile_dir: str = "~/Spotlight/default/",
    visualize_detections: bool = False,
    num_workers: int = -1,
) -> None:
    """
    Fit homography transformation between behavior and muscle cameras using ChArUco board.
    
    Args:
        profile_dir: Path to the profile directory
        visualize_detections: Whether to visualize ChArUco detections
        num_workers: Number of parallel workers (-1 for all CPUs, 1 for no parallelization)
    """
    print("Fitting homography between behavior and muscle cameras...")
    
    # Expand profile directory
    profile_dir = Path(profile_dir).expanduser()
    
    # Check if calibration images exist
    calibration_image_dir = profile_dir / "calibration/charuco_homography_scan"
    if not calibration_image_dir.exists():
        raise FileNotFoundError(
            f"Directory {calibration_image_dir} does not exist. "
            "Run calibration scan first."
        )
    
    # Create output directories
    output_dir = profile_dir / "calibration/model/homography"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualization_dir = None
    if visualize_detections:
        visualization_dir = profile_dir / "calibration/charuco_detection"
        visualization_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize ChArUco board
    charuco_board = get_gizem_board()
    
    # Gather matching corners
    print("Detecting ChArUco corners in image pairs...")
    matching_data = gather_matching_corners(
        calibration_image_dir,
        visualization_dir=visualization_dir,
        num_workers=num_workers,
    )
    
    if len(matching_data) == 0:
        raise ValueError("No matching corners found in any image pair!")
    
    print(f"Found {len(matching_data)} image pairs with matching corners")
    
    # Compute homography
    print("Computing homography transformation...")
    H_beh2muscle, inliers, behavior_pts, muscle_pts = compute_homography_from_matches(matching_data)
    
    # Compute inverse homography
    H_muscle2beh = np.linalg.inv(H_beh2muscle)
    
    # Visualize quality
    quality_viz_path = output_dir / "homography_quality.png"
    visualize_homography_quality(H_beh2muscle, inliers, behavior_pts, muscle_pts, quality_viz_path)
    
    # Save results to YAML
    homography_results = {
        "metadata": {
            "file_format_version": {
                "major": versions.calibration_result_major,
                "minor": versions.calibration_result_minor,
                "patch": versions.calibration_result_patch,
            },
            "description": "Homography transformation between behavior and muscle cameras",
            "num_image_pairs": len(matching_data),
            "num_matching_points": len(behavior_pts),
            "num_inliers": int(np.sum(inliers)),
            "inlier_percentage": float(100 * np.sum(inliers) / len(behavior_pts)),
        },
        "behavior_to_muscle": {
            "matrix": [
                [float(H_beh2muscle[0, 0]), float(H_beh2muscle[0, 1]), float(H_beh2muscle[0, 2])],
                [float(H_beh2muscle[1, 0]), float(H_beh2muscle[1, 1]), float(H_beh2muscle[1, 2])],
                [float(H_beh2muscle[2, 0]), float(H_beh2muscle[2, 1]), float(H_beh2muscle[2, 2])],
            ],
            "description": "Homography matrix to transform behavior camera coordinates to muscle camera coordinates",
        },
        "muscle_to_behavior": {
            "matrix": [
                [float(H_muscle2beh[0, 0]), float(H_muscle2beh[0, 1]), float(H_muscle2beh[0, 2])],
                [float(H_muscle2beh[1, 0]), float(H_muscle2beh[1, 1]), float(H_muscle2beh[1, 2])],
                [float(H_muscle2beh[2, 0]), float(H_muscle2beh[2, 1]), float(H_muscle2beh[2, 2])],
            ],
            "description": "Homography matrix to transform muscle camera coordinates to behavior camera coordinates",
        },
    }
    
    results_path = output_dir / "homography_result.yaml"
    with open(results_path, "w") as f:
        yaml.dump(homography_results, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nHomography results saved to {results_path}")
    print(f"Inlier ratio: {homography_results['metadata']['inlier_percentage']:.1f}%")
    print("\nHomography matrix (behavior -> muscle):")
    print(H_beh2muscle)


def main():
    tyro.cli(fit_homography)


if __name__ == "__main__":
    main()