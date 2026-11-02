"""Global localization of a robot against the room in room.py.

The room is the source of truth. One scan, taken while the robot is still,
is matched against the room's walls to find the rigid 2D transform on the
server's X/Z ground plane that carries the robot's odometry frame onto the
room. That transform becomes the robot's coordinate system offset.

Matching runs in two passes:
  1. Coarse: a correlative search over every yaw and every robot position
     inside the room. The scan's outer bounds are scored against a wall
     likelihood field, so walls count, clutter inside the room is neutral,
     and points beyond the walls (which a lidar cannot see) are penalised.
  2. Fine: the best few distinct coarse peaks are refined with ICP against
     points sampled along the walls and rescored. Furniture can make a wrong
     pose win the coarse pass, so the winner is only picked after refining.
"""

import math
from dataclasses import dataclass
from functools import cache

import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

from map.icp import icp
from map.room import ROOM_CORNERS

# Coarse search cell size and yaw step.
GRID_RESOLUTION = 0.1
YAW_STEP = math.radians(1.0)

# Only the farthest point in each angular sector around the robot is kept, so
# the scan is reduced to its outer bounds.
SECTOR_WIDTH = math.radians(1.0)

# Wall likelihood falloff, and the penalty for a point well beyond a wall.
# Only an open door lets the lidar see past a wall, so the penalty is steep.
WALL_SIGMA = 0.15
OUTSIDE_TOLERANCE = 0.3
OUTSIDE_PENALTY = 4.0

# The robot's centre must be at least this far inside the walls.
ROBOT_RADIUS = 0.2

# A point within this distance of a wall after refinement is an inlier.
INLIER_DISTANCE = 0.1
MIN_INLIER_RATIO = 0.4
MIN_POINTS = 30

# Coarse peaks closer than this in both position and yaw count as one pose.
# The best CANDIDATES distinct poses are refined.
DISTINCT_DISTANCE = 1.0
DISTINCT_YAW = math.radians(20.0)
CANDIDATES = 8

# A distinct refined pose scoring this close to the best one means the scan
# cannot tell them apart. OUTSIDE_PENALTY and this were tuned on simulated
# scans with furniture: no wrong placements in 150 poses, 15% rejected.
MAX_AMBIGUITY = 0.85

WALL_SAMPLE_SPACING = 0.02


class LocalizationError(Exception):
    """The scan could not be placed in the room."""


@dataclass(frozen=True)
class RoomLocalization:
    # Homogeneous 2D transform taking odometry-frame X/Z onto room X/Z.
    matrix: np.ndarray
    # Fractions of the scan's outer bounds on a wall and beyond the walls.
    inlier_ratio: float
    outside_ratio: float
    # Runner-up score over best score; near 1 means another pose fits as well.
    ambiguity: float

    @property
    def yaw(self) -> float:
        """Rotation from X toward Z, i.e. counterclockwise seen from above."""
        return math.atan2(self.matrix[1, 0], self.matrix[0, 0])

    @property
    def origin(self) -> tuple[float, float]:
        """Room X/Z of the odometry frame's origin."""
        return float(self.matrix[0, 2]), float(self.matrix[1, 2])


def _rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _translation(x: float, z: float) -> np.ndarray:
    return np.array([[1.0, 0.0, x], [0.0, 1.0, z], [0.0, 0.0, 1.0]])


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:2, :2].T + matrix[:2, 2]


class _RoomModel:
    """Wall likelihood field and wall samples for one room outline."""

    def __init__(self, corners: tuple[tuple[float, float], ...]):
        self.corners = np.array(corners, dtype=float)
        lo = self.corners.min(axis=0)
        hi = self.corners.max(axis=0)
        self.x0, self.z0 = lo
        self.width = int(math.ceil((hi[0] - lo[0]) / GRID_RESOLUTION)) + 1
        self.height = int(math.ceil((hi[1] - lo[1]) / GRID_RESOLUTION)) + 1

        # No scan point inside the room can be farther from the robot than
        # the room's diagonal, plus a little for points seen through doors.
        self.max_range = float(np.hypot(*(hi - lo))) + 1.0
        self.pad = int(math.ceil(self.max_range / GRID_RESOLUTION)) + 2

        # Rows are Z and columns are X, offset by the padding.
        shift = 4
        pixels = np.round(
            ((self.corners - lo) / GRID_RESOLUTION + self.pad) * (1 << shift)
        ).astype(np.int32)
        shape = (self.height + 2 * self.pad, self.width + 2 * self.pad)

        inside = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(inside, [pixels], 1, shift=shift)
        walls = np.full(shape, 255, dtype=np.uint8)
        cv2.polylines(walls, [pixels], True, 0, shift=shift)
        wall_distance = cv2.distanceTransform(walls, cv2.DIST_L2, 5) * GRID_RESOLUTION

        field = np.exp(-(wall_distance ** 2) / (2 * WALL_SIGMA ** 2))
        field[(inside == 0) & (wall_distance > OUTSIDE_TOLERANCE)] = -OUTSIDE_PENALTY
        self.field = field.astype(np.float32)

        # Where the robot's centre may be, on the unpadded room grid.
        p = self.pad
        self.robot_allowed = (
            (inside[p:p + self.height, p:p + self.width] == 1)
            & (wall_distance[p:p + self.height, p:p + self.width] >= ROBOT_RADIUS)
        )

        samples = []
        for start, end in zip(self.corners, np.roll(self.corners, -1, axis=0)):
            count = max(int(np.linalg.norm(end - start) / WALL_SAMPLE_SPACING), 1)
            t = np.arange(count)[:, None] / count
            samples.append(start + t * (end - start))
        self.wall_points = np.vstack(samples)
        self.wall_neighbors = NearestNeighbors(n_neighbors=1).fit(self.wall_points)
        self.polygon = self.corners.astype(np.float32).reshape(-1, 1, 2)

    def fit(self, world: np.ndarray) -> tuple[float, float]:
        """Return the fractions of points on a wall and beyond the walls."""
        distances, _ = self.wall_neighbors.kneighbors(world)
        distances = distances[:, 0]
        on_wall = distances < INLIER_DISTANCE
        outside = np.array([
            cv2.pointPolygonTest(self.polygon, (float(x), float(z)), False) < 0
            for x, z in world
        ]) & (distances > OUTSIDE_TOLERANCE)
        return float(np.mean(on_wall)), float(np.mean(outside))

    def cell_position(self, row: int, col: int) -> tuple[float, float]:
        return self.x0 + col * GRID_RESOLUTION, self.z0 + row * GRID_RESOLUTION

    def score_yaw(self, relative: np.ndarray, yaw: float) -> np.ndarray:
        """Mean wall likelihood of the scan for every robot cell at one yaw.

        relative holds the scan points relative to the robot, in odometry X/Z.
        """
        rotated = _apply(_rotation(yaw), relative)
        cols = np.round(rotated[:, 0] / GRID_RESOLUTION).astype(int)
        rows = np.round(rotated[:, 1] / GRID_RESOLUTION).astype(int)
        # The robot's own cell is always part of the template.
        row_min, col_min = min(rows.min(), 0), min(cols.min(), 0)
        robot_row, robot_col = -row_min, -col_min
        template = np.zeros(
            (max(rows.max(), 0) - row_min + 1, max(cols.max(), 0) - col_min + 1),
            dtype=np.float32,
        )
        np.add.at(template, (rows - row_min, cols - col_min), 1.0)

        # Crop the field so result[row, col] is the robot at room cell (row, col).
        top = self.pad - robot_row
        left = self.pad - robot_col
        image = self.field[
            top:top + self.height - 1 + template.shape[0],
            left:left + self.width - 1 + template.shape[1],
        ]
        scores = cv2.matchTemplate(image, template, cv2.TM_CCORR) / len(relative)
        return np.where(self.robot_allowed, scores, -np.inf)


@cache
def _room_model(corners: tuple[tuple[float, float], ...]) -> _RoomModel:
    return _RoomModel(corners)


def outer_bounds(relative: np.ndarray) -> np.ndarray:
    """Return the indices of the farthest point in each sector around the robot."""
    if len(relative) == 0:
        return np.zeros(0, dtype=int)
    ranges = np.hypot(relative[:, 0], relative[:, 1])
    sectors = np.floor(
        (np.arctan2(relative[:, 1], relative[:, 0]) + math.pi) / SECTOR_WIDTH
    ).astype(int)
    # Sort by sector, then range, and keep the last (farthest) of each sector.
    order = np.lexsort((ranges, sectors))
    last = np.append(sectors[order][1:] != sectors[order][:-1], True)
    return order[last]


def localize_in_room(points: np.ndarray, robot: np.ndarray) -> RoomLocalization:
    """Place a scan in the room.

    points: (N, 2) scan points in the odometry frame's X/Z, using the server's
            axis directions but none of the robot's offset.
    robot:  (2,) robot position in the same frame when the scan was taken.

    Raises LocalizationError if the scan cannot be placed confidently.
    """
    room = _room_model(tuple(map(tuple, ROOM_CORNERS)))

    relative = points - robot
    relative = relative[np.hypot(relative[:, 0], relative[:, 1]) <= room.max_range]
    relative = relative[outer_bounds(relative)]
    if len(relative) < MIN_POINTS:
        raise LocalizationError(f"only {len(relative)} usable scan points")

    yaws = np.arange(0.0, 2 * math.pi, YAW_STEP)
    scores = np.stack([room.score_yaw(relative, yaw) for yaw in yaws])

    # Local maxima over yaw (which wraps) and position.
    neighborhood = np.stack([cv2.dilate(layer, np.ones((3, 3), np.uint8)) for layer in scores])
    neighborhood = np.maximum(neighborhood, np.maximum(np.roll(neighborhood, 1, 0), np.roll(neighborhood, -1, 0)))
    peaks = np.flatnonzero((scores == neighborhood) & np.isfinite(scores) & (scores > 0))
    if len(peaks) == 0:
        raise LocalizationError("scan does not match any wall")

    # Greedily keep the best peaks that are distinct poses.
    candidates: list[tuple[float, float, float]] = []
    for index in peaks[np.argsort(scores.flat[peaks])[::-1]]:
        yaw_index, row, col = np.unravel_index(index, scores.shape)
        yaw = float(yaws[yaw_index])
        x, z = room.cell_position(row, col)
        if not any(_same_pose((yaw, x, z), other) for other in candidates):
            candidates.append((yaw, x, z))
            if len(candidates) == CANDIDATES:
                break

    # Refine each candidate and rescore it.
    scan = relative + robot
    refined: list[tuple[float, np.ndarray, float, float]] = []
    for yaw, x, z in candidates:
        coarse = _translation(x, z) @ _rotation(yaw) @ _translation(-robot[0], -robot[1])
        history, _ = icp(room.wall_points, _apply(coarse, scan))
        matrix = coarse
        for step in history:
            matrix = np.vstack((step, [0.0, 0.0, 1.0])) @ matrix
        inlier_ratio, outside_ratio = room.fit(_apply(matrix, scan))
        refined.append((inlier_ratio - OUTSIDE_PENALTY * outside_ratio, matrix, inlier_ratio, outside_ratio))
    refined.sort(key=lambda candidate: candidate[0], reverse=True)

    best_score, matrix, inlier_ratio, outside_ratio = refined[0]
    best_pose = _pose(matrix, robot)
    # ICP can pull two candidates onto the same pose, so compare final poses.
    runner_up = max(
        (score for score, other, _, _ in refined[1:] if not _same_pose(_pose(other, robot), best_pose)),
        default=0.0,
    )
    ambiguity = max(runner_up, 0.0) / best_score if best_score > 0 else 1.0

    if inlier_ratio < MIN_INLIER_RATIO:
        raise LocalizationError(f"only {inlier_ratio:.0%} of the scan lies on the room's walls")
    if ambiguity > MAX_AMBIGUITY:
        raise LocalizationError(
            f"scan fits more than one place in the room (runner-up scores {ambiguity:.0%} of best)"
        )

    return RoomLocalization(
        matrix=matrix,
        inlier_ratio=inlier_ratio,
        outside_ratio=outside_ratio,
        ambiguity=ambiguity,
    )


def _pose(matrix: np.ndarray, robot: np.ndarray) -> tuple[float, float, float]:
    """(yaw, x, z) of the robot in the room under matrix."""
    x, z = _apply(matrix, robot[None])[0]
    return math.atan2(matrix[1, 0], matrix[0, 0]), float(x), float(z)


def _same_pose(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    yaw_gap = abs((a[0] - b[0] + math.pi) % (2 * math.pi) - math.pi)
    return yaw_gap < DISTINCT_YAW and math.hypot(a[1] - b[1], a[2] - b[2]) < DISTINCT_DISTANCE
