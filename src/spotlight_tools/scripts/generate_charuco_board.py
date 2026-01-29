import tyro
from pathlib import Path
from spotlight_tools.calibration.charuco import ChArUcoBoard


def generate_charuco(
    arena_size_x_mm: float = 48,
    arena_size_y_mm: float = 72,
    output_path: str = "~/Spotlight/charuco_board/charuco_board.svg",
):
    """Generate an ArUco board with the specified parameters.

    Args:
        arena_size_x_mm (float):
            The width of the arena in mm.
        arena_size_y_mm (float):
            The height of the arena in mm.
        output_path (str):
            The path to save the generated SVG file.
    """
    aruco_board = ChArUcoBoard(
        arena_dim_mm=(arena_size_x_mm, arena_size_y_mm),
    )
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aruco_board.draw_svg(output_path)
    aruco_board.draw_png(output_path.with_suffix(".png"))


def main():
    tyro.cli(generate_charuco)


if __name__ == "__main__":
    main()
