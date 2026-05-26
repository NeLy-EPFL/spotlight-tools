"""
spotlight_tools.arena -- arena configuration and registration fitting.

ArenaConfig
    Parses a PDF arena spec (arena146_spec.pdf) to extract AprilTag and
    DataMatrix positions, rasterise the active-area mask, and write the
    three derived assets (mapping_board.pdf, active_area.png, metadata.yaml)
    to the arena directory.

fit_arena_registration
    End-to-end pipeline: reads the `mapping_scan/` images produced by
    `run-arena-registration-scan`, detects AprilTags, rejects outliers via
    MAD filtering and RANSAC, and writes calibration_result.yaml plus
    diagnostic plots to `<arena_dir>/model/`.
"""

from .arena import ArenaConfig
from .registration import (
    APRILTAG_FAMILY,
    detect_apriltags,
    filter_per_apriltag_outliers,
    fit_arena_registration,
    fit_ransac_model,
    gather_apriltag_points,
)
