import cv2
import numpy as np


def draw_vector_overlay(
    frame: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    h_field: int,
    w_field: int,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Draw vector field overlay on a frame using arrow visualization.

    Args:
        frame: Input image frame
        u: Horizontal component of vector field
        v: Vertical component of vector field
        grid_x: X coordinates of grid points
        grid_y: Y coordinates of grid points
        h_field: Height dimension of the field
        w_field: Width dimension of the field
        scale: Scaling factor for vector magnitude

    Returns:
        Frame with vector overlay drawn
    """
    for i in range(h_field):
        for j in range(w_field):
            dx = float(u[i, j]) * scale
            dy = float(v[i, j]) * scale
            start_point = (int(grid_x[i, j]), int(grid_y[i, j]))
            end_point = (int(start_point[0] + dx), int(start_point[1] + dy))
            cv2.arrowedLine(
                frame,
                start_point,
                end_point,
                color=(0, 255, 0),
                thickness=3,
                tipLength=0.3,
            )
    return frame


def draw_scalar_overlay(frame: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    """
    Draw scalar field overlay on a frame using heatmap visualization.

    Args:
        frame: Input image frame
        scalar: Scalar field data to visualize

    Returns:
        Frame with scalar heatmap overlay
    """
    resized = cv2.resize(scalar.numpy(), (frame.shape[1], frame.shape[0]))
    normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
