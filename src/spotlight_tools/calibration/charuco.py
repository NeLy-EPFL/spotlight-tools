import cv2
import numpy as np
import matplotlib.pyplot as plt
import svgwrite

def get_opencv_charuco_detection_params():
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.minMarkers = 0
    charuco_params.tryRefineMarkers = True
    return charuco_params

def _preprocess_image(image, camera):
    if camera == "behavior_camera":
        # For behavior camera images, apply a binary threshold and morphological closing
        thr = cv2.threshold(image, 75, 255, cv2.THRESH_BINARY)[1]
    elif camera == "muscle_camera":
        # For muscle camera images, apply a binary threshold and morphological closing
        thr = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1]
    cleaned = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8), iterations=1)
    return cleaned.copy()

def detect_charuco(
    image, board, camera, horizontal_flip=False
):
    """
    Detect ArUco codes in an image and return their IDs and corner coordinates.

    Parameters:
    -----------
    image : numpy.ndarray
        Grayscale image of shape (rows, cols)
    board : cv2.aruco.CharucoBoard
        The Charuco board object to use for detection.
    horizontal_flip : bool, optional
        Whether to flip the image horizontally before detection (default: False)

    Returns:
    --------
    tuple
        (ids, coords) where:
        - ids is an array of shape (n, 1) corners ids
        - coords is an array of shape (n, 1, 2) containing the x, y coordinates
        of each detected corner
    """
    # Preprocess image
    working_image = _preprocess_image(image, camera)
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].imshow(image, cmap="gray")
    # axs[0].set_title("Original Image")
    # axs[1].imshow(working_image, cmap="gray")
    # axs[1].set_title("Preprocessed Image")
    # plt.show(block=True)

    # Get image dimensions
    num_rows, num_cols = working_image.shape

    # Flip the image horizontally if requested
    if horizontal_flip:
        working_image = cv2.flip(working_image, 1)  # 1 means horizontal flip

    charuco_detection_params = get_opencv_charuco_detection_params()
    detector = cv2.aruco.CharucoDetector(board.board, charucoParams=charuco_detection_params)
    charuco_corners, charuco_corner_ids, _, _ = detector.detectBoard(working_image)

    # If no markers are detected, return empty arrays
    if charuco_corner_ids is None:
        return np.zeros((0, 1)), np.zeros((0, 2))
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    charuco_corners = cv2.cornerSubPix(
        working_image,
        charuco_corners,
        winSize=(71, 71),
        zeroZone=(-1, -1),
        criteria=criteria,
    )

    # Convert charuco corners to proper format (squeeze unnecessary dimensions)
    coords = charuco_corners
    # Flip coordinates back if the image was flipped
    if horizontal_flip:
        # For each detected corner
        for i in range(coords.shape[0]):
            # Flip the x-coordinate
            coords[i, 0, 0] = num_cols - coords[i, 0, 0]

    return charuco_corner_ids.flatten(), coords


def plot_charuco_detections(fig, ax, image, ids, coords):
    """
    Plot the detected Charuco codes on the image with different colors for each corner.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure object to plot on
    ax : matplotlib.axes.Axes
        Axes object to plot on
    image : numpy.ndarray
        Original grayscale image
    ids : numpy.ndarray
        Array of ArUco IDs of shape (n,)
    coords : numpy.ndarray
        Array of corner coordinates of shape (n, 1, 2)
    """
    # Display the image
    ax.imshow(image, cmap="gray")

    # Colors for the four corners
    corner_color = "cyan"

    # For each detected ArUco marker
    for i in range(len(ids)):
        ax.scatter(
            coords[i, 0, 0],
            coords[i, 0, 1],
            c=corner_color,
            s=5,
            marker="x",
        )
        ax.text(
            coords[i, 0, 0] + 5,
            coords[i, 0, 1] - 5,
            str(ids[i]),
            color="yellow",
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )

    # Set title and labels
    ax.set_title(f"Detected {len(ids)} ArUco Markers")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    plt.tight_layout()


class ChArUcoBoard:
    square_length_mm = 2.4 # Size of one checkerboard square
    min_margin_mm = 2.0  # Minimum margin around the checkerboard
    id_offset = 1 # Offset in counting corners
    marker_length_mm = 1.8  # Size of one ArUco marker
    code_size_unitblk = 6
    scale_mm = marker_length_mm / code_size_unitblk  # Size of one unit block in mm

    def __init__(self, arena_dim_mm):
        self.arena_dim_mm = arena_dim_mm  # (width, height)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        n_codes_h = int(
            (self.arena_dim_mm[0] - 2 * self.min_margin_mm) / self.square_length_mm
        )
        n_codes_v = int(
            (self.arena_dim_mm[1] - 2 * self.min_margin_mm) / self.square_length_mm
        )
        self.margin = (
            (self.arena_dim_mm[0] -n_codes_h * self.square_length_mm) / 2,
            (self.arena_dim_mm[1] - n_codes_v * self.square_length_mm) / 2,
        )

        # Calculate how many codes fit in each dimension
        self.grid_dim_ncodes = (
            n_codes_h,
            n_codes_v,
        )

        self.board = cv2.aruco.CharucoBoard(
            (self.grid_dim_ncodes[0], self.grid_dim_ncodes[1]),
            self.square_length_mm,
            self.marker_length_mm,
            self.aruco_dict,
        )

        # Check if we have enough markers
        ncodes_total = (self.grid_dim_ncodes[0] * self.grid_dim_ncodes[1]) / 2
        if ncodes_total > self.aruco_dict.bytesList.shape[0]:
            raise ValueError(
                "Not enough markers in ArUco dictionary for the given arena size"
            )
        
    def corner_id_to_row_col(self, corner_id):
        """
        nbr_corners_perdim = nbr_codes + 1
        nbr_detectable_corners_perdim = nbr_corners_perdim - 2
        => nbr_corners_perdim = codes - 1
        """
        row = corner_id // (self.grid_dim_ncodes[0]-1)
        col = corner_id % (self.grid_dim_ncodes[0]-1)
        return row, col
    
    def corner_id_to_xy_mm(self, corner_id):
        id_row, id_col = self.corner_id_to_row_col(corner_id)
        x = self.margin[0] + ((id_col + self.id_offset) * self.square_length_mm)
        y = self.margin[1] + ((id_row + self.id_offset) * self.square_length_mm)
        return x, y

    def draw_png(self, output_path):
        
        dpi=600
        dpmm = dpi / 25.4  # Convert dpi to dots per mm
        size = [int(val*dpmm) for val in self.arena_dim_mm]
        img = self.board.generateImage(
            size,
            marginSize=int(self.min_margin_mm * dpmm),
        )
        cv2.imwrite(output_path, img)

    def draw_svg(self, output_path):
        dwg = svgwrite.Drawing(
            output_path,
            profile="tiny",
            size=(
                f"{self.arena_dim_mm[0]}mm",  # width
                f"{self.arena_dim_mm[1]}mm",  # height
            ),
        )

        self.marker_margin = (self.square_length_mm - self.marker_length_mm) / 2

        # Iterate through the grid and generate ArUco codes
        for r in range(self.grid_dim_ncodes[1]):
            for c in range(self.grid_dim_ncodes[0]):
                # Generate a unique ArUco marker ID
                checker_id = (r * self.grid_dim_ncodes[0] + c)
                if (r+c) % 2 == 0:
                    # draw a black square for even checker IDs
                    x_offset_mm = self.margin[0] + c * self.square_length_mm
                    y_offset_mm = self.margin[1] + r * self.square_length_mm
                    dwg.add(
                        dwg.rect(
                            insert=(f"{x_offset_mm}mm", f"{y_offset_mm}mm"),
                            size=(f"{self.square_length_mm}mm", f"{self.square_length_mm}mm"),
                            fill="black",
                        )
                    )
                else:
                    # For odd checker IDs, draw the ArUco marker
                    marker_id = checker_id // 2  # Integer division to get marker ID

                    # Create the marker image
                    marker_img = np.zeros(
                        (self.code_size_unitblk, self.code_size_unitblk), dtype=np.uint8
                    )
                    cv2.aruco.generateImageMarker(
                        self.aruco_dict, marker_id, self.code_size_unitblk, marker_img, 1
                    )

                    # Calculate the position of the top-left corner of the marker
                    x_offset_mm = self.margin[0] + c * self.square_length_mm + self.marker_margin
                    y_offset_mm = self.margin[1] + r * self.square_length_mm + self.marker_margin

                    # Add the marker to the SVG as a group of rectangles
                    for px_row in range(self.code_size_unitblk):
                        for px_col in range(self.code_size_unitblk):
                            if marker_img[px_row, px_col] == 0:  # Draw black squares
                                dwg.add(
                                    dwg.rect(
                                        insert=(
                                            f"{x_offset_mm + px_col * self.scale_mm}mm",
                                            f"{y_offset_mm + px_row * self.scale_mm}mm",
                                        ),
                                        size=(f"{self.scale_mm}mm", f"{self.scale_mm}mm"),
                                        fill="black",
                                    )
                                )
        dwg.save()
