from typing import Optional, List
import cv2
import supervision as sv
import numpy as np
from sports.configs.soccer import SoccerPitchConfiguration

def _scale_and_pad_points(points: np.ndarray, scale: float, padding: int) -> np.ndarray:
    """Scale and pad points for pitch drawing."""
    return np.array([(int(p[0] * scale) + padding, int(p[1] * scale) + padding) for p in points])

def _validate_inputs(config: SoccerPitchConfiguration, scale: float, padding: int, xy: Optional[np.ndarray] = None):
    """Validate common input parameters."""
    if scale <= 0:
        raise ValueError("Scale must be positive.")
    if padding < 0:
        raise ValueError("Padding must be non-negative.")
    if not isinstance(config, SoccerPitchConfiguration):
        raise TypeError("Config must be a SoccerPitchConfiguration object.")
    if xy is not None and (xy.ndim != 2 or xy.shape[1] != 2):
        raise ValueError("xy must be a 2D NumPy array with shape (n, 2).")

def draw_pitch(
    config: SoccerPitchConfiguration,
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1
) -> np.ndarray:
    """
    Draws a soccer pitch with specified dimensions, colors, and scale.
    
    Args:
        config: SoccerPitchConfiguration object with pitch dimensions and layout.
        background_color: Color of the pitch background (default: green).
        line_color: Color of the pitch lines (default: white).
        padding: Padding around the pitch in pixels (default: 50).
        line_thickness: Thickness of the pitch lines in pixels (default: 4).
        point_radius: Radius of penalty spot points in pixels (default: 8).
        scale: Scaling factor for pitch dimensions (default: 0.1).

    Returns:
        np.ndarray: Image of the soccer pitch (shape: (height, width, 3), dtype: uint8).

    Raises:
        ValueError: If scale or padding is invalid.
    """
    _validate_inputs(config, scale, padding)
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    pitch_image = np.ones((scaled_width + 2 * padding, scaled_length + 2 * padding, 3), dtype=np.uint8) * np.array(background_color.as_bgr(), dtype=np.uint8)

    try:
        for start, end in config.edges:
            point1, point2 = _scale_and_pad_points(np.array([config.vertices[start - 1], config.vertices[end - 1]]), scale, padding)
            cv2.line(pitch_image, tuple(point1), tuple(point2), line_color.as_bgr(), line_thickness)

        centre_circle_center = (scaled_length // 2 + padding, scaled_width // 2 + padding)
        cv2.circle(pitch_image, centre_circle_center, scaled_circle_radius, line_color.as_bgr(), line_thickness)

        penalty_spots = [
            (scaled_penalty_spot_distance + padding, scaled_width // 2 + padding),
            (scaled_length - scaled_penalty_spot_distance + padding, scaled_width // 2 + padding)
        ]
        for spot in penalty_spots:
            cv2.circle(pitch_image, spot, point_radius, line_color.as_bgr(), -1)
    except IndexError as e:
        raise IndexError("Invalid vertex indices in config.edges or config.vertices.") from e

    return pitch_image

def draw_points_on_pitch(
    config: SoccerPitchConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws points on a soccer pitch.

    Args:
        config: SoccerPitchConfiguration object with pitch dimensions and layout.
        xy: Array of (x, y) coordinates for points to draw (shape: (n, 2)).
        face_color: Color of point faces (default: red).
        edge_color: Color of point edges (default: black).
        radius: Radius of points in pixels (default: 10).
        thickness: Thickness of point edges in pixels (default: 2).
        padding: Padding around the pitch in pixels (default: 50).
        scale: Scaling factor for pitch dimensions (default: 0.1).
        pitch: Existing pitch image to draw points on (default: None).

    Returns:
        np.ndarray: Image of the soccer pitch with points (shape: (height, width, 3), dtype: uint8).

    Raises:
        ValueError: If inputs are invalid.
    """
    _validate_inputs(config, scale, padding, xy)
    if pitch is None:
        pitch = draw_pitch(config, padding=padding, scale=scale)

    try:
        scaled_points = _scale_and_pad_points(xy, scale, padding)
        for point in scaled_points:
            cv2.circle(pitch, tuple(point), radius, face_color.as_bgr(), -1)
            cv2.circle(pitch, tuple(point), radius, edge_color.as_bgr(), thickness)
    except Exception as e:
        raise RuntimeError("Error drawing points on pitch.") from e

    return pitch

def draw_paths_on_pitch(
    config: SoccerPitchConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws paths on a soccer pitch.

    Args:
        config: SoccerPitchConfiguration object with pitch dimensions and layout.
        paths: List of arrays, each containing (x, y) coordinates for a path.
        color: Color of the paths (default: white).
        thickness: Thickness of paths in pixels (default: 2).
        padding: Padding around the pitch in pixels (default: 50).
        scale: Scaling factor for pitch dimensions (default: 0.1).
        pitch: Existing pitch image to draw paths on (default: None).

    Returns:
        np.ndarray: Image of the soccer pitch with paths (shape: (height, width, 3), dtype: uint8).

    Raises:
        ValueError: If inputs are invalid.
    """
    _validate_inputs(config, scale, padding)
    if not paths or not all(isinstance(path, np.ndarray) and path.ndim == 2 and path.shape[1] == 2 for path in paths):
        raise ValueError("Paths must be a list of 2D NumPy arrays with shape (n, 2).")
    
    if pitch is None:
        pitch = draw_pitch(config, padding=padding, scale=scale)

    try:
        for path in paths:
            scaled_path = _scale_and_pad_points(path, scale, padding)
            if len(scaled_path) < 2:
                continue
            for i in range(len(scaled_path) - 1):
                cv2.line(pitch, tuple(scaled_path[i]), tuple(scaled_path[i + 1]), color.as_bgr(), thickness)
    except Exception as e:
        raise RuntimeError("Error drawing paths on pitch.") from e

    return pitch

def draw_pitch_voronoi_diagram(
    config: SoccerPitchConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.RED,
    team_2_color: sv.Color = sv.Color.WHITE,
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws a Voronoi diagram on a soccer pitch for two teams' control areas.

    Args:
        config: SoccerPitchConfiguration object with pitch dimensions and layout.
        team_1_xy: Array of (x, y) coordinates for team 1 players (shape: (n, 2)).
        team_2_xy: Array of (x, y) coordinates for team 2 players (shape: (m, 2)).
        team_1_color: Color for team 1 control area (default: red).
        team_2_color: Color for team 2 control area (default: white).
        opacity: Opacity of the Voronoi diagram overlay (default: 0.5).
        padding: Padding around the pitch in pixels (default: 50).
        scale: Scaling factor for pitch dimensions (default: 0.1).
        pitch: Existing pitch image to draw Voronoi diagram on (default: None).

    Returns:
        np.ndarray: Image of the soccer pitch with Voronoi diagram (shape: (height, width, 3), dtype: uint8).

    Raises:
        ValueError: If inputs are invalid.
    """
    _validate_inputs(config, scale, padding, team_1_xy)
    _validate_inputs(config, scale, padding, team_2_xy)
    if not 0 <= opacity <= 1:
        raise ValueError("Opacity must be between 0 and 1.")

    if pitch is None:
        pitch = draw_pitch(config, padding=padding, scale=scale)

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    voronoi = np.zeros_like(pitch, dtype=np.uint8)

    try:
        # Vectorized distance calculation
        y_coords, x_coords = np.indices((scaled_width + 2 * padding, scaled_length + 2 * padding))
        coords = np.stack([x_coords - padding, y_coords - padding], axis=-1).reshape(-1, 2)
        
        team_1_points = team_1_xy * scale
        team_2_points = team_2_xy * scale
        
        # Compute distances for all points at once
        def calculate_distances(points):
            return np.sqrt(np.sum((points[:, None, :] - coords[None, :, :]) ** 2, axis=-1))
        
        distances_team_1 = calculate_distances(team_1_points)
        distances_team_2 = calculate_distances(team_2_points)
        
        min_distances_team_1 = np.min(distances_team_1, axis=0).reshape(scaled_width + 2 * padding, scaled_length + 2 * padding)
        min_distances_team_2 = np.min(distances_team_2, axis=0).reshape(scaled_width + 2 * padding, scaled_length + 2 * padding)
        
        control_mask = min_distances_team_1 < min_distances_team_2
        voronoi[control_mask] = np.array(team_1_color.as_bgr(), dtype=np.uint8)
        voronoi[~control_mask] = np.array(team_2_color.as_bgr(), dtype=np.uint8)
        
        overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)
    except Exception as e:
        raise RuntimeError("Error computing Voronoi diagram.") from e

    return overlay
