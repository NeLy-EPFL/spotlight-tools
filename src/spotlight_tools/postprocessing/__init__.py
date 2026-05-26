"""
spotlight_tools.postprocessing -- full offline processing pipeline for recordings.

Modules
-------
behavior
    Decode pseudo-BGR JPEGs, run SLEAP 2-D pose estimation, and produce
    aligned / cropped behavior video.  Entry point:
    decode_and_align_all_behavior_frames.
muscle
    Warp muscle frames into the behavior-camera coordinate system and apply
    the same fly-alignment transforms.  Entry point:
    warp_all_muscle_frames_to_behavior.
stage
    Interpolate stage XY positions at each behavior-frame timestamp.
    Entry point: interp_stage_pos_at_behavior_frames.
visualize
    Generate summary video, 2-D pose overlays, muscle-behavior overlay
    samples, and stage-trajectory plot.
io
    Low-level helpers: find per-frame files, check output path consistency.
"""

from .behavior import (
    decode_and_align_all_behavior_frames,
    expand_single_pseudo_bgr_image,
    estimate_2dpose_sequence,
    fill_gaps_in_2dpose_sequence,
    transform_single_frame_to_align,
)
from .muscle import (
    warp_single_muscle_frame_to_behavior,
    warp_all_muscle_frames_to_behavior,
    match_behavior_frameid_to_muscle_frameid,
    match_muscle_frameid_to_behavior_frameid,
)
from .stage import interp_stage_pos_at_behavior_frames
from .io import find_files_per_frame_by_suffix
from .visualize import (
    visualize_stage_trajectory,
    generate_summary_video,
    draw_2dpose_on_single_frame,
    generate_overlay_samples,
)
