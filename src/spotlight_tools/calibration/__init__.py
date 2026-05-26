"""
spotlight_tools.calibration -- legacy ArUco-based calibration (still used by postprocessing).

This package implements the original ArUco board scanning and linear mapper
that was used before the arena146 AprilTag registration pipeline. It is
retained because postprocessing/muscle.py still depends on
SpotlightPositionMapper and BehaviorMuscleCrossMapper to warp muscle frames
into the behavior camera coordinate system.

For the current arena registration workflow, use spotlight_tools.arena instead.
"""

from .aruco import ArUcoBoard, detect_aruco, plot_aruco_detections
from .charuco import CharucoBoard, get_gizem_board
from .model import ransac_filter_outliers, visualize_ransac_results
from .mapper import SpotlightPositionMapper, BehaviorMuscleCrossMapper, HomographyMapper
