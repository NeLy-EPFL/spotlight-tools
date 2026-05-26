"""
spotlight_tools.common -- shared utilities used across the package.

config
    load_spotlight_tools_config(): reads config/config.yaml from the package
    root (SLEAP model path, keypoint names, etc.).
video
    get_video_info, get_video_writer, write_video: thin wrappers around
    OpenCV/vidgear for reading and writing encoded video files.
"""

from .config import load_spotlight_tools_config
from .video import get_video_info, write_video, get_video_writer
