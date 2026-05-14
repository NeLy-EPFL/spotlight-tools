"""
To generate the mapping boards for the two standard Spotlight arenas:

    generate-mapping-board \
        --arena-width-mm 48 \
        --arena-height-mm 72 \
        --margin-mm 5 \
        --apriltag-pixel-size-mm 0.6 \
        --aztec-pixel-size-mm 0.2 \
        --output-path mapping_board_48x72.pdf

    generate-mapping-board \
        --arena-width-mm 146 \
        --arena-height-mm 146 \
        --margin-mm 5 \
        --apriltag-pixel-size-mm 0.6 \
        --aztec-pixel-size-mm 0.2 \
        --output-path mapping_board_146x146.pdf
"""

from pathlib import Path

import tyro

from spotlight_tools.calibration.mapping_board import EightMarkersMappingBoard


def generate_mapping_board(
    arena_width_mm: float,
    arena_height_mm: float,
    margin_mm: float,
    output_path: str,
    apriltag_pixel_size_mm: float = 0.6,
    aztec_pixel_size_mm: float = 0.2,
):
    """Generate a mapping board with the specified parameters.

    Args:
        arena_width_mm: The width of the arena in mm.
        arena_height_mm: The height of the arena in mm.
        margin_mm: Distance from the outer edge of the marker to the arena
            boundary in mm.
        output_path: Path to save the generated PDF file.
        apriltag_pixel_size_mm: Side length of one AprilTag pixel in mm.
        aztec_pixel_size_mm: Side length of one Aztec module in mm.
    """
    board = EightMarkersMappingBoard(
        arena_dim_mm=(arena_width_mm, arena_height_mm),
        apriltag_pixel_size_mm=apriltag_pixel_size_mm,
        aztec_pixel_size_mm=aztec_pixel_size_mm,
        margin_mm=margin_mm,
    )
    board.draw_pdf(Path(output_path))


def main():
    tyro.cli(generate_mapping_board)


if __name__ == "__main__":
    main()