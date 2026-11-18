"""BSP-based floor plan generator for HVAC simulations.

Generates 2D floor plans using Binary Space Partitioning algorithm.

Internal grid encoding: 0 = floor/empty, 1 = wall, 2 = door.
Exported simulator format (matching constants.py):
    0 = interior space (INTERIOR_SPACE_VALUE_IN_FILE_INPUT)
    1 = walls (INTERIOR_WALL_VALUE_IN_FILE_INPUT)
    2 = exterior (EXTERIOR_SPACE_VALUE_IN_FILE_INPUT)
With 5-cell exterior padding around the perimeter.
"""

import math
import random

from smart_buildings.smart_control.reward.energy_scale import (
    C_ELECTRICITY_PER_AHU,
    C_NATURAL_GAS_PER_AHU,
)
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage


def _crop_to_content(
    grid: np.ndarray, padding: int = 5, exterior: float = 2.0
) -> np.ndarray:
    """Trim rows and columns that are nothing but exterior, keeping a margin.

    A floor plan is generated on a rectangular canvas, but a real footprint
    rarely fills it: fill runs 57-90% across the scenario set. Everything the
    building does not occupy is still exterior cells the solver sweeps over on
    every step, so the empty border is paid for on all 8640 steps of an episode
    and contributes nothing.

    Cropping to the occupied bounding box plus `padding` is exact, not an
    approximation. The removed cells are exterior before and absent after, and
    the boundary condition is unchanged because `padding` cells of exterior are
    kept on every side.
    """
    occupied = grid != exterior
    if not occupied.any():
        return grid
    rows = np.flatnonzero(occupied.any(axis=1))
    cols = np.flatnonzero(occupied.any(axis=0))
    r0 = max(0, rows[0] - padding)
    r1 = min(grid.shape[0], rows[-1] + 1 + padding)
    c0 = max(0, cols[0] - padding)
    c1 = min(grid.shape[1], cols[-1] + 1 + padding)
    return grid[r0:r1, c0:c1]


@dataclass
class BSPNode:
    x: int
    y: int
    width: int
    height: int
    left: Optional["BSPNode"] = None
    right: Optional["BSPNode"] = None
    room: tuple[int, int, int, int] | None = None


class BSPFloorPlan:
    """Generate floor plans using Binary Space Partitioning."""

    SHAPES = [
        "rectangle",
        "trapezoid",
        "h_shape",
        "t_shape",
        "pentagon",
        "oval",
        "u_shape",
        "parallelogram",
        "semicircle",
        "triangle",
        "composite",
    ]

    # Base shapes for random composite (exclude composite itself to avoid recursion)
    BASE_SHAPES = [
        "rectangle",
        "trapezoid",
        "h_shape",
        "t_shape",
        "pentagon",
        "oval",
        "u_shape",
        "parallelogram",
        "semicircle",
        "triangle",
    ]

    # Axis-aligned shapes used by the architectural composite generator
    ARCHITECTURAL_SHAPES = [
        "rectangle",
        "trapezoid",
        "h_shape",
        "t_shape",
        "u_shape",
    ]

    COMPOSITE_STYLES = ["architectural", "random"]
    MAX_COMPOSITE_COMPACTNESS = 45.0

    def __init__(
        self,
        width=50,
        height=50,
        min_room_size=6,
        max_room_size=15,
        wall_thickness=1,
        door_width=2,
        split_variance=0.3,
        building_shape="rectangle",
        shape_ratio=0.33,
        composite_coverage=0.2,
        composite_style="architectural",
        predefined_mask=None,
        notch_fill_cells=4,
    ):
        """
        Args:
            width: Grid width in cells
            height: Grid height in cells
            min_room_size: Minimum room dimension
            max_room_size: Maximum room dimension before forcing split
            wall_thickness: Thickness of exterior walls
            door_width: Width of doors between rooms
            split_variance: Randomness in split position (0-1)
            building_shape: One of SHAPES
            shape_ratio: For H/T/U shapes, controls thick vs thin parts (0.1 to 0.5)
                - H shape: width of vertical bars as fraction of total width
                - T shape: height of top bar & width of stem as fraction
                - U shape: width of side bars & height of bottom as fraction
            composite_coverage: For composite shape, minimum canvas coverage (0.0 to 1.0)
            composite_style: "architectural" (wing-based) or "random" (legacy union)
            predefined_mask: Optional pre-defined shape mask to use instead of generating one
            notch_fill_cells: Width of edge notch, in cells, that composite
                post-processing closes up. Keep it near the size of the artefact
                (a few cells); scaling it to the room size flattens the massing.
        """
        self.width = width
        self.height = height
        self.min_room_size = min_room_size
        self.max_room_size = max_room_size
        self.wall_thickness = wall_thickness
        self.door_width = door_width
        self.split_variance = split_variance
        self.building_shape = building_shape.lower()
        self.shape_ratio = max(0.1, min(0.5, shape_ratio))
        self.composite_coverage = max(0.0, min(1.0, composite_coverage))
        style = composite_style.lower()
        if style not in self.COMPOSITE_STYLES:
            raise ValueError(
                f"composite_style must be one of {self.COMPOSITE_STYLES}, got {composite_style!r}"
            )
        self.composite_style = style
        self.predefined_mask = predefined_mask
        self.notch_fill_cells = max(0, int(notch_fill_cells))

    def generate(self) -> np.ndarray:
        """Generate floor plan grid.

        Returns:
            np.ndarray with 0=floor, 1=wall, 2=door
        """
        self.mask = self._create_shape_mask()
        self.interior = self._erode_mask(self.mask, self.wall_thickness)
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        outer_wall = self.mask & ~self.interior
        self.grid[outer_wall] = 1
        root = BSPNode(0, 0, self.width, self.height)
        self._split(root)
        self._add_internal_walls(root)
        self._extend_floating_walls()
        self._connect_rooms(root)
        return self.grid

    def _erode_mask(self, mask, iterations):
        result = mask.copy()
        for _ in range(iterations):
            new_result = np.zeros_like(result)
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    if (
                        result[y, x]
                        and result[y - 1, x]
                        and result[y + 1, x]
                        and result[y, x - 1]
                        and result[y, x + 1]
                    ):
                        new_result[y, x] = True
            result = new_result
        return result

    def _create_shape_mask(self):
        # Use predefined mask if provided
        if self.predefined_mask is not None:
            return self.predefined_mask.copy()

        mask = np.zeros((self.height, self.width), dtype=bool)
        h, w = self.height, self.width
        r = self.shape_ratio

        if self.building_shape == "rectangle":
            mask[:, :] = True

        elif self.building_shape == "trapezoid":
            for y in range(h):
                t = y / max(1, h - 1)
                indent = int((1 - t) * w * 0.25)
                mask[y, indent : w - indent] = True

        elif self.building_shape == "h_shape":
            bar_width = max(3, int(w * r))
            connector_height = max(3, int(h * r))
            mid_y = h // 2
            mask[:, :bar_width] = True
            mask[:, w - bar_width :] = True
            mask[
                mid_y - connector_height // 2 : mid_y + connector_height // 2 + 1, :
            ] = True

        elif self.building_shape == "t_shape":
            top_height = max(3, int(h * r))
            stem_width = max(3, int(w * r))
            stem_start = (w - stem_width) // 2
            mask[:top_height, :] = True
            mask[top_height:, stem_start : stem_start + stem_width] = True

        elif self.building_shape == "pentagon":
            cx = w // 2
            for y in range(h):
                for x in range(w):
                    if y > h * 0.3:
                        if abs(x - cx) < w * 0.45:
                            mask[y, x] = True
                    else:
                        if abs(x - cx) < (w * 0.45) / (h * 0.3) * y:
                            mask[y, x] = True

        elif self.building_shape == "oval":
            cx, cy = w // 2, h // 2
            rx, ry = w // 2 - 1, h // 2 - 1
            for y in range(h):
                for x in range(w):
                    if ((x - cx) / max(1, rx)) ** 2 + ((y - cy) / max(1, ry)) ** 2 <= 1:
                        mask[y, x] = True

        elif self.building_shape == "u_shape":
            bar_width = max(3, int(w * r))
            bottom_height = max(3, int(h * r))
            mask[h - bottom_height :, :] = True
            mask[:, :bar_width] = True
            mask[:, w - bar_width :] = True

        elif self.building_shape == "parallelogram":
            skew = w // 4
            for y in range(h):
                t = y / max(1, h - 1)
                offset = int(t * skew)
                x_start, x_end = offset, w - skew + offset
                if x_start < w and x_end > 0:
                    mask[y, max(0, x_start) : min(w, x_end)] = True

        elif self.building_shape == "semicircle":
            cx = w // 2
            radius = min(w // 2, h) - 1
            for y in range(h):
                for x in range(w):
                    if ((x - cx) ** 2 + y**2) ** 0.5 <= radius:
                        mask[y, x] = True

        elif self.building_shape == "triangle":
            cx = w // 2
            for y in range(h):
                t = y / max(1, h - 1)
                half_width = int(t * w / 2)
                if half_width > 0:
                    mask[y, cx - half_width : cx + half_width] = True
                elif y == 0:
                    mask[y, cx] = True

        elif self.building_shape == "composite":
            mask = self._create_composite_mask()

        else:
            mask[:, :] = True
        return mask

    def _create_composite_mask(self) -> np.ndarray:
        """Create a composite footprint with post-processing and quality checks."""
        max_attempts = 5
        for _ in range(max_attempts):
            if self.composite_style == "random":
                mask = self._create_random_composite_mask()
            else:
                mask = self._create_architectural_composite_mask()
            mask = self._postprocess_composite_mask(mask)
            if self._mask_passes_quality_gate(mask):
                return mask

        fallback_shape = "rectangle"
        mask = self._create_single_shape_mask(self.width, self.height, fallback_shape)
        return self._postprocess_composite_mask(mask)

    def _create_random_composite_mask(self) -> np.ndarray:
        """Legacy composite: overlay random shapes until coverage threshold is met."""
        h, w = self.height, self.width
        mask = np.zeros((h, w), dtype=bool)
        target_coverage = h * w * self.composite_coverage
        min_scale = 0.3
        max_scale = 0.6
        max_iterations = 50
        iterations = 0

        while np.sum(mask) < target_coverage and iterations < max_iterations:
            shape_name = random.choice(self.BASE_SHAPES)
            scale = random.uniform(min_scale, max_scale)
            sub_w = max(10, int(w * scale))
            sub_h = max(10, int(h * scale))
            max_x = w - sub_w // 2
            max_y = h - sub_h // 2
            min_x = -sub_w // 2
            min_y = -sub_h // 2
            pos_x = random.randint(min_x, max_x)
            pos_y = random.randint(min_y, max_y)
            sub_mask = self._create_single_shape_mask(sub_w, sub_h, shape_name)
            mask = self._union_sub_mask(mask, sub_mask, pos_x, pos_y)
            iterations += 1

        return mask

    def _create_architectural_composite_mask(self) -> np.ndarray:
        """Build a composite from a centered base block plus anchored wings."""
        h, w = self.height, self.width
        mask = np.zeros((h, w), dtype=bool)

        base_w = max(self.min_room_size * 2, int(w * random.uniform(0.55, 0.75)))
        base_h = max(self.min_room_size * 2, int(h * random.uniform(0.55, 0.75)))
        base_x = (w - base_w) // 2
        base_y = (h - base_h) // 2
        mask[base_y : base_y + base_h, base_x : base_x + base_w] = True

        num_wings = random.randint(1, 3)
        directions = random.sample(["north", "south", "east", "west"], num_wings)
        for direction in directions:
            mask = self._attach_composite_wing(mask, direction)

        target_coverage = h * w * self.composite_coverage
        expand_attempts = 0
        while np.sum(mask) < target_coverage and expand_attempts < 8:
            direction = random.choice(["north", "south", "east", "west"])
            mask = self._attach_composite_wing(mask, direction)
            expand_attempts += 1

        mask = self._expand_mask_to_coverage(mask)
        return mask

    def _expand_mask_to_coverage(self, mask: np.ndarray) -> np.ndarray:
        """Grow the footprint evenly from its bounds until coverage target is met."""
        h, w = self.height, self.width
        target_coverage = h * w * self.composite_coverage
        if np.sum(mask) >= target_coverage:
            return mask

        y_min, y_max, x_min, x_max = self._get_mask_bounds(mask)
        max_iters = max(h, w)

        for _ in range(max_iters):
            if np.sum(mask) >= target_coverage:
                break

            grown = False
            if y_min > 0:
                mask[y_min - 1, x_min : x_max + 1] = True
                y_min -= 1
                grown = True
            if y_max < h - 1:
                mask[y_max + 1, x_min : x_max + 1] = True
                y_max += 1
                grown = True
            if x_min > 0:
                mask[y_min : y_max + 1, x_min - 1] = True
                x_min -= 1
                grown = True
            if x_max < w - 1:
                mask[y_min : y_max + 1, x_max + 1] = True
                x_max += 1
                grown = True

            if not grown:
                break

        return mask

    def _get_mask_bounds(self, mask: np.ndarray) -> tuple[int, int, int, int]:
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return 0, self.height - 1, 0, self.width - 1
        return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())

    def _union_sub_mask(
        self, mask: np.ndarray, sub_mask: np.ndarray, pos_x: int, pos_y: int
    ) -> np.ndarray:
        sub_h, sub_w = sub_mask.shape
        for sy in range(sub_h):
            ty = pos_y + sy
            if ty < 0 or ty >= self.height:
                continue
            sx_start = max(0, -pos_x)
            sx_end = min(sub_w, self.width - pos_x)
            if sx_start >= sx_end:
                continue
            tx_start = pos_x + sx_start
            tx_end = pos_x + sx_end
            mask[ty, tx_start:tx_end] |= sub_mask[sy, sx_start:sx_end]
        return mask

    def _attach_composite_wing(self, mask: np.ndarray, direction: str) -> np.ndarray:
        h, w = self.height, self.width
        y_min, y_max, x_min, x_max = self._get_mask_bounds(mask)
        shape_name = random.choice(["rectangle", "rectangle", "h_shape", "t_shape", "u_shape"])
        overlap = random.randint(2, 4)

        if direction in ("east", "west"):
            span = y_max - y_min + 1
            sub_h = max(
                self.min_room_size,
                min(span, int(h * random.uniform(0.25, 0.5))),
            )
            sub_w = max(self.min_room_size, int(w * random.uniform(0.15, 0.35)))
            pos_y = max(0, min(h - sub_h, (y_min + y_max - sub_h) // 2))
            if direction == "east":
                pos_x = min(w - sub_w, x_max - overlap + 1)
            else:
                pos_x = max(0, x_min - sub_w + overlap)
        else:
            span = x_max - x_min + 1
            sub_w = max(
                self.min_room_size,
                min(span, int(w * random.uniform(0.25, 0.5))),
            )
            sub_h = max(self.min_room_size, int(h * random.uniform(0.15, 0.35)))
            pos_x = max(0, min(w - sub_w, (x_min + x_max - sub_w) // 2))
            if direction == "south":
                pos_y = min(h - sub_h, y_max - overlap + 1)
            else:
                pos_y = max(0, y_min - sub_h + overlap)

        sub_mask = self._create_single_shape_mask(sub_w, sub_h, shape_name)
        return self._union_sub_mask(mask, sub_mask, pos_x, pos_y)

    def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        labeled, num_labels = ndimage.label(mask)
        if num_labels <= 1:
            return mask
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        largest_label = int(np.argmax(counts))
        return labeled == largest_label

    def _postprocess_composite_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = self._keep_largest_component(mask)

        # Overlapping wings leave two kinds of artefact that are not rooms and
        # cannot be occupied: fully enclosed voids a metre or so across, and
        # narrow rectangular notches bitten out of the edge. Both survive
        # _keep_largest_component, which only picks the largest foreground blob
        # and says nothing about the background inside it.
        mask = ndimage.binary_fill_holes(mask)

        # Closing fills a concave notch only if the notch is narrower than the
        # structuring element. The radius has to stay small: scaled to the room
        # grain it reaches 15-20 cells, which swallows the wings themselves and
        # flattens a composite plan into a plain rectangle. Sized instead to the
        # defect, a few cells, it removes the slits and leaves the massing.
        # iterations=0 does not mean "skip" in scipy; it erases the mask, so the
        # disable case has to return before the call rather than pass 0 through.
        if self.notch_fill_cells > 0:
            mask = ndimage.binary_closing(
                mask, np.ones((3, 3), dtype=bool), iterations=self.notch_fill_cells
            )

        # Closing can bridge two wings that merely pass close to one another;
        # re-select so the result stays a single building.
        return self._keep_largest_component(mask)

    def _mask_compactness(self, mask: np.ndarray) -> float:
        area = int(np.sum(mask))
        if area == 0:
            return float("inf")
        perimeter = 0
        padded = np.pad(mask, 1, constant_values=False)
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if not mask[y, x]:
                    continue
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if not padded[y + 1 + dy, x + 1 + dx]:
                        perimeter += 1
        return (perimeter * perimeter) / area

    def _mask_passes_quality_gate(self, mask: np.ndarray) -> bool:
        if not np.any(mask):
            return False
        labeled, num_labels = ndimage.label(mask)
        if num_labels != 1:
            return False
        target_coverage = self.height * self.width * self.composite_coverage
        if np.sum(mask) < target_coverage * 0.95:
            return False
        erosion_iters = max(1, self.min_room_size // 2)
        struct = np.ones((3, 3), dtype=bool)
        eroded = mask.copy()
        for _ in range(erosion_iters):
            eroded = ndimage.binary_erosion(eroded, structure=struct)
        if not np.any(eroded):
            return False
        return self._mask_compactness(mask) <= self.MAX_COMPOSITE_COMPACTNESS

    def _create_single_shape_mask(self, w: int, h: int, shape_name: str) -> np.ndarray:
        """Create a single shape mask at the specified dimensions."""
        mask = np.zeros((h, w), dtype=bool)
        r = self.shape_ratio

        if shape_name == "rectangle":
            mask[:, :] = True

        elif shape_name == "trapezoid":
            for y in range(h):
                t = y / max(1, h - 1)
                indent = int((1 - t) * w * 0.25)
                mask[y, indent : w - indent] = True

        elif shape_name == "h_shape":
            bar_width = max(2, int(w * r))
            connector_height = max(2, int(h * r))
            mid_y = h // 2
            mask[:, :bar_width] = True
            mask[:, w - bar_width :] = True
            mask[
                max(0, mid_y - connector_height // 2) : min(
                    h, mid_y + connector_height // 2 + 1
                ),
                :,
            ] = True

        elif shape_name == "t_shape":
            top_height = max(2, int(h * r))
            stem_width = max(2, int(w * r))
            stem_start = (w - stem_width) // 2
            mask[:top_height, :] = True
            mask[top_height:, stem_start : stem_start + stem_width] = True

        elif shape_name == "pentagon":
            cx = w // 2
            for y in range(h):
                for x in range(w):
                    if y > h * 0.3:
                        if abs(x - cx) < w * 0.45:
                            mask[y, x] = True
                    else:
                        if abs(x - cx) < (w * 0.45) / max(1, h * 0.3) * y:
                            mask[y, x] = True

        elif shape_name == "oval":
            cx, cy = w // 2, h // 2
            rx, ry = max(1, w // 2 - 1), max(1, h // 2 - 1)
            for y in range(h):
                for x in range(w):
                    if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1:
                        mask[y, x] = True

        elif shape_name == "u_shape":
            bar_width = max(2, int(w * r))
            bottom_height = max(2, int(h * r))
            mask[h - bottom_height :, :] = True
            mask[:, :bar_width] = True
            mask[:, w - bar_width :] = True

        elif shape_name == "parallelogram":
            skew = w // 4
            for y in range(h):
                t = y / max(1, h - 1)
                offset = int(t * skew)
                x_start, x_end = offset, w - skew + offset
                if x_start < w and x_end > 0:
                    mask[y, max(0, x_start) : min(w, x_end)] = True

        elif shape_name == "semicircle":
            cx = w // 2
            radius = min(w // 2, h) - 1
            for y in range(h):
                for x in range(w):
                    if ((x - cx) ** 2 + y**2) ** 0.5 <= radius:
                        mask[y, x] = True

        elif shape_name == "triangle":
            cx = w // 2
            for y in range(h):
                t = y / max(1, h - 1)
                half_width = int(t * w / 2)
                if half_width > 0:
                    mask[y, max(0, cx - half_width) : min(w, cx + half_width)] = True
                elif y == 0:
                    mask[y, cx] = True

        else:
            mask[:, :] = True

        return mask

    def _region_has_space(self, x, y, w, h):
        if x < 0 or y < 0 or x + w > self.width or y + h > self.height:
            return False
        return np.sum(self.interior[y : y + h, x : x + w]) >= (w * h * 0.3)

    def _split(self, node, depth=0):
        if node.width < self.min_room_size * 2 or node.height < self.min_room_size * 2:
            return
        if max(node.width, node.height) < self.max_room_size and random.random() < 0.3:
            return
        if node.width > node.height * 1.25:
            horizontal = False
        elif node.height > node.width * 1.25:
            horizontal = True
        else:
            horizontal = random.random() < 0.5
        if horizontal:
            max_split = node.height - self.min_room_size
            min_split = self.min_room_size
            if max_split <= min_split:
                return
            variance = int((max_split - min_split) * self.split_variance)
            split = node.height // 2 + random.randint(-variance, variance)
            split = max(min_split, min(max_split, split))
            left = BSPNode(node.x, node.y, node.width, split)
            right = BSPNode(node.x, node.y + split, node.width, node.height - split)
        else:
            max_split = node.width - self.min_room_size
            min_split = self.min_room_size
            if max_split <= min_split:
                return
            variance = int((max_split - min_split) * self.split_variance)
            split = node.width // 2 + random.randint(-variance, variance)
            split = max(min_split, min(max_split, split))
            left = BSPNode(node.x, node.y, split, node.height)
            right = BSPNode(node.x + split, node.y, node.width - split, node.height)
        if self._region_has_space(left.x, left.y, left.width, left.height):
            node.left = left
        if self._region_has_space(right.x, right.y, right.width, right.height):
            node.right = right
        if node.left:
            self._split(node.left, depth + 1)
        if node.right:
            self._split(node.right, depth + 1)

    def _add_internal_walls(self, node):
        if node.left is None and node.right is None:
            node.room = (node.x, node.y, node.width, node.height)
            return
        if node.left:
            self._add_internal_walls(node.left)
        if node.right:
            self._add_internal_walls(node.right)
        if node.left and node.right:
            if node.left.y == node.right.y:
                wall_x = node.left.x + node.left.width
                for y in range(node.y, node.y + node.height):
                    if (
                        0 <= y < self.height
                        and 0 <= wall_x < self.width
                        and self.mask[y, wall_x]
                    ):
                        self.grid[y, wall_x] = 1
            else:
                wall_y = node.left.y + node.left.height
                for x in range(node.x, node.x + node.width):
                    if (
                        0 <= wall_y < self.height
                        and 0 <= x < self.width
                        and self.mask[wall_y, x]
                    ):
                        self.grid[wall_y, x] = 1

    def _extend_floating_walls(self):
        """Post-process to extend walls that end in thin air (v2 fix)."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        changed = True
        while changed:
            changed = False
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    if self.grid[y, x] != 1:
                        continue
                    is_perimeter = False
                    for dy, dx in directions:
                        if not self.mask[y + dy, x + dx]:
                            is_perimeter = True
                            break
                    if is_perimeter:
                        continue
                    wall_neighbors = [
                        (dy, dx)
                        for dy, dx in directions
                        if self.grid[y + dy, x + dx] == 1
                    ]
                    if len(wall_neighbors) == 1:
                        wy, wx = wall_neighbors[0]
                        ey, ex = y - wy, x - wx
                        if self.mask[ey, ex] and self.grid[ey, ex] == 0:
                            self.grid[ey, ex] = 1
                            changed = True

    def _get_room(self, node):
        if node.room:
            return node.room
        left_room = self._get_room(node.left) if node.left else None
        right_room = self._get_room(node.right) if node.right else None
        if left_room and right_room:
            return random.choice([left_room, right_room])
        return left_room or right_room

    def _connect_rooms(self, node):
        if node.left is None or node.right is None:
            return
        self._connect_rooms(node.left)
        self._connect_rooms(node.right)
        if node.left.y == node.right.y:
            wall_x = node.left.x + node.left.width
            valid_ys = [
                y
                for y in range(node.y + 1, node.y + node.height - 1)
                if 0 <= y < self.height
                and 0 <= wall_x < self.width
                and self.interior[y, wall_x]
            ]
            if valid_ys:
                door_y = valid_ys[len(valid_ys) // 2]
                for dy in range(self.door_width):
                    if 0 <= door_y + dy < self.height:
                        self.grid[door_y + dy, wall_x] = 2
        else:
            wall_y = node.left.y + node.left.height
            valid_xs = [
                x
                for x in range(node.x + 1, node.x + node.width - 1)
                if 0 <= wall_y < self.height
                and 0 <= x < self.width
                and self.interior[wall_y, x]
            ]
            if valid_xs:
                door_x = valid_xs[len(valid_xs) // 2]
                for dx in range(self.door_width):
                    if 0 <= door_x + dx < self.width:
                        self.grid[wall_y, door_x + dx] = 2

    def export_for_simulator(self, padding: int = 5, crop: bool = True) -> np.ndarray:
        """Export floor plan in HVAC simulator format.

        Simulator format (from constants.py):
            0 = interior space (floor/rooms) - INTERIOR_SPACE_VALUE_IN_FILE_INPUT
            1 = walls - INTERIOR_WALL_VALUE_IN_FILE_INPUT
            2 = exterior space (outside building) - EXTERIOR_SPACE_VALUE_IN_FILE_INPUT

        Args:
            padding: Number of exterior cells to add around the perimeter (default 5)

        Returns:
            np.ndarray with simulator-compatible encoding
        """
        # Create grid with padding - default to exterior (2)
        padded_height = self.height + 2 * padding
        padded_width = self.width + 2 * padding
        sim_grid = np.full(
            (padded_height, padded_width), 2.0, dtype=np.float64
        )  # default: exterior = 2

        # Map internal grid to simulator format within the padded region
        for y in range(self.height):
            for x in range(self.width):
                py, px = y + padding, x + padding
                if not self.mask[y, x]:
                    # Outside building shape = exterior
                    sim_grid[py, px] = 2
                elif self.grid[y, x] == 1:
                    # Walls
                    sim_grid[py, px] = 1
                else:
                    # Interior space (floor and doors)
                    sim_grid[py, px] = 0

        return _crop_to_content(sim_grid, padding) if crop else sim_grid

    def export_zone_map(self, padding: int = 5, crop: bool = True) -> np.ndarray:
        """Export zone map for room detection (doors as walls).

        Zone map format (from constants.py):
            0 = interior space (floor only, no doors) - INTERIOR_SPACE_VALUE_IN_FILE_INPUT
            1 = walls AND doors (barriers for zone detection) - INTERIOR_WALL_VALUE_IN_FILE_INPUT
            2 = exterior space (outside building) - EXTERIOR_SPACE_VALUE_IN_FILE_INPUT

        This ensures each room is detected as a separate zone.

        Args:
            padding: Number of exterior cells to add around the perimeter (default 5)

        Returns:
            np.ndarray with zone detection encoding
        """
        # Create grid with padding - default to exterior (2)
        padded_height = self.height + 2 * padding
        padded_width = self.width + 2 * padding
        zone_map = np.full(
            (padded_height, padded_width), 2.0, dtype=np.float64
        )  # default: exterior = 2

        # Map internal grid to zone map format within the padded region
        for y in range(self.height):
            for x in range(self.width):
                py, px = y + padding, x + padding
                if not self.mask[y, x]:
                    # Outside building shape = exterior
                    zone_map[py, px] = 2
                elif self.grid[y, x] == 1 or self.grid[y, x] == 2:
                    # Walls and doors = barriers for zone detection
                    zone_map[py, px] = 1
                else:
                    # Interior space (floor only)
                    zone_map[py, px] = 0

        return _crop_to_content(zone_map, padding) if crop else zone_map

    def save_for_simulator(
        self,
        floor_plan_path: str,
        zone_map_path: str | None = None,
        padding: int = 5,
    ) -> tuple[str, str]:
        """Save floor plan and zone map as .npy files for HVAC simulator.

        Args:
            floor_plan_path: Path to save floor plan (doors=floor for airflow)
            zone_map_path: Path to save zone map (doors=walls for zone detection)
                          If None, uses floor_plan_path with '_zones' suffix
            padding: Number of exterior cells to add around the perimeter (default 5)

        Returns:
            Tuple of (floor_plan_path, zone_map_path)
        """
        if zone_map_path is None:
            base = floor_plan_path.replace(".npy", "")
            zone_map_path = f"{base}_zones.npy"

        np.save(floor_plan_path, self.export_for_simulator(padding=padding))
        np.save(zone_map_path, self.export_zone_map(padding=padding))
        return floor_plan_path, zone_map_path

    def get_num_rooms(self, padding: int = 5) -> int:
        """Count the number of rooms using connected components.

        Uses connected components on zone_map (doors as walls).
        Room labels match simulator: room_1, room_2, etc. in raster order.

        Args:
            padding: Number of exterior cells around the perimeter (default 5)
        """
        import cv2

        zone_map = self.export_zone_map(padding=padding)
        # zone_map: 0 = interior space, 1 = walls, 2 = exterior
        binary = (zone_map == 0).astype(np.uint8)
        num_labels, _ = cv2.connectedComponents(binary, connectivity=4)
        return num_labels - 1

    def get_room_labels(self, padding: int = 5) -> tuple[np.ndarray, list[str]]:
        """Get room labels matching simulator naming convention.

        Args:
            padding: Number of exterior cells around the perimeter (default 5)

        Returns:
            Tuple of (labeled_grid, room_names)
            - labeled_grid: array where each room has unique integer label
            - room_names: list of room names in order ['room_1', 'room_2', ...]
        """
        import cv2

        zone_map = self.export_zone_map(padding=padding)
        # zone_map: 0 = interior space, 1 = walls, 2 = exterior
        binary = (zone_map == 0).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary, connectivity=4)
        room_names = [f"room_{i}" for i in range(1, num_labels)]
        return labels, room_names


class MultiFloorPlan:
    """Generate multi-floor building by stacking single-floor plans vertically.

    Same API as BSPFloorPlan, plus num_floors parameter.
    All dimension parameters (width, height, room sizes) refer to a single floor.
    """

    def __init__(
        self,
        width=50,
        height=50,
        num_floors=1,
        min_room_size=6,
        max_room_size=15,
        wall_thickness=1,
        door_width=2,
        split_variance=0.3,
        building_shape="rectangle",
        shape_ratio=0.33,
        composite_coverage=0.2,
        composite_style="architectural",
    ):
        """
        Args:
            width: Grid width in cells (per floor)
            height: Grid height in cells (per floor)
            num_floors: Number of floors to stack vertically
            min_room_size: Minimum room dimension
            max_room_size: Maximum room dimension before forcing split
            wall_thickness: Thickness of exterior walls
            door_width: Width of doors between rooms
            split_variance: Randomness in split position (0-1)
            building_shape: One of BSPFloorPlan.SHAPES
            shape_ratio: For H/T/U shapes, controls thick vs thin parts (0.1 to 0.5)
            composite_coverage: For composite shape, minimum canvas coverage (0.0 to 1.0)
            composite_style: "architectural" (wing-based) or "random" (legacy union)
        """
        self.width = width
        self.height = height
        self.num_floors = num_floors
        self.min_room_size = min_room_size
        self.max_room_size = max_room_size
        self.wall_thickness = wall_thickness
        self.door_width = door_width
        self.split_variance = split_variance
        self.building_shape = building_shape.lower()
        self.shape_ratio = max(0.1, min(0.5, shape_ratio))
        self.composite_coverage = max(0.0, min(1.0, composite_coverage))
        style = composite_style.lower()
        if style not in BSPFloorPlan.COMPOSITE_STYLES:
            raise ValueError(
                f"composite_style must be one of {BSPFloorPlan.COMPOSITE_STYLES}, got {composite_style!r}"
            )
        self.composite_style = style

        self.floors: list[BSPFloorPlan] = []
        self.grid: np.ndarray | None = None
        self.mask: np.ndarray | None = None

    def generate(self) -> np.ndarray:
        """Generate floor plan grid with all floors stacked vertically.

        Returns:
            np.ndarray with 0=floor, 1=wall, 2=door (floors stacked vertically,
            separated by 10 pixels of air)
        """
        self.floors = []
        floor_grids = []
        floor_masks = []

        # Separator: 10 rows of empty space between floors
        separator_grid = np.zeros((10, self.width), dtype=np.int8)
        separator_mask = np.zeros((10, self.width), dtype=bool)

        # For consistent outlines across floors, capture first floor's mask
        shared_mask = None

        for i in range(self.num_floors):
            floor = BSPFloorPlan(
                width=self.width,
                height=self.height,
                min_room_size=self.min_room_size,
                max_room_size=self.max_room_size,
                wall_thickness=self.wall_thickness,
                door_width=self.door_width,
                split_variance=self.split_variance,
                building_shape=self.building_shape,
                shape_ratio=self.shape_ratio,
                composite_coverage=self.composite_coverage,
                composite_style=self.composite_style,
                predefined_mask=shared_mask,  # Use shared mask for floors after first
            )
            floor.generate()

            # Capture first floor's mask to reuse for all subsequent floors
            if i == 0:
                shared_mask = floor.mask

            self.floors.append(floor)
            floor_grids.append(floor.grid)
            floor_masks.append(floor.mask)
            # Add separator after each floor except the last
            if i < self.num_floors - 1:
                floor_grids.append(separator_grid)
                floor_masks.append(separator_mask)

        self.grid = np.vstack(floor_grids)
        self.mask = np.vstack(floor_masks)
        return self.grid

    def export_for_simulator(self, padding: int = 5) -> np.ndarray:
        """Export floor plan in HVAC simulator format.

        Returns:
            np.ndarray with simulator-compatible encoding (floors stacked vertically,
            separated by 10 pixels of exterior space)
        """
        if not self.floors:
            raise ValueError("Must call generate() first")

        # Separator: 10 rows of exterior space (value 2)
        padded_width = self.width + 2 * padding
        separator = np.full((10, padded_width), 2.0, dtype=np.float64)

        result = []
        for i, floor in enumerate(self.floors):
            result.append(floor.export_for_simulator(padding=padding, crop=False))
            if i < self.num_floors - 1:
                result.append(separator)

        # Crop once, after stacking. Cropping per floor would give each floor a
        # different width and the fixed-width separator would no longer fit.
        return _crop_to_content(np.vstack(result), padding)

    def export_zone_map(self, padding: int = 5) -> np.ndarray:
        """Export zone map for room detection (doors as walls).

        Returns:
            np.ndarray with zone detection encoding (floors stacked vertically,
            separated by 10 pixels of exterior space)
        """
        if not self.floors:
            raise ValueError("Must call generate() first")

        # Separator: 10 rows of exterior space (value 2)
        padded_width = self.width + 2 * padding
        separator = np.full((10, padded_width), 2.0, dtype=np.float64)

        result = []
        for i, floor in enumerate(self.floors):
            result.append(floor.export_zone_map(padding=padding, crop=False))
            if i < self.num_floors - 1:
                result.append(separator)

        # Crop once, after stacking. Cropping per floor would give each floor a
        # different width and the fixed-width separator would no longer fit.
        return _crop_to_content(np.vstack(result), padding)

    def save_for_simulator(
        self,
        floor_plan_path: str,
        zone_map_path: str | None = None,
        padding: int = 5,
    ) -> tuple[str, str]:
        """Save floor plan and zone map as .npy files for HVAC simulator."""
        if zone_map_path is None:
            base = floor_plan_path.replace(".npy", "")
            zone_map_path = f"{base}_zones.npy"

        np.save(floor_plan_path, self.export_for_simulator(padding=padding))
        np.save(zone_map_path, self.export_zone_map(padding=padding))
        return floor_plan_path, zone_map_path

    def get_num_rooms(self, padding: int = 5) -> int:
        """Count total number of rooms in the stacked floor plan.

        Uses connected components on the full stacked zone map to match
        how the simulator discovers rooms.
        """
        import cv2

        if not self.floors:
            raise ValueError("Must call generate() first")
        zone_map = self.export_zone_map(padding=padding)
        binary = (zone_map == 0).astype(np.uint8)
        num_labels, _ = cv2.connectedComponents(binary, connectivity=4)
        return num_labels - 1  # Subtract background label

    def get_room_labels(self, padding: int = 5) -> tuple[np.ndarray, list[str]]:
        """Get room labels for the stacked floor plan.

        Returns:
            Tuple of (labeled_grid, room_names)
            - labeled_grid: array where each room has unique integer label
            - room_names: list of room names ['room_1', 'room_2', ...]
        """
        import cv2

        zone_map = self.export_zone_map(padding=padding)
        binary = (zone_map == 0).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary, connectivity=4)
        room_names = [f"room_{i}" for i in range(1, num_labels)]
        return labels, room_names


def required_ahus(num_rooms: int, max_rooms_per_ahu: int) -> int:
    """Fewest air handlers that can serve `num_rooms` without oversubscription.

    An air handler moves `AHUConfig.max_air_flow_rate` and each VAV can draw
    `vav_max_air_flowrate`, so one handler can serve
    `max_air_flow_rate / vav_max_air_flowrate` zones with every damper open at
    once. At the shipped 8.67 and 2.0 that is 4.33 zones, and the measured
    saturation curve agrees exactly: air handlers serving 2, 3 or 4 zones never
    ran out of air over 864 steps, while 5 zones saturated 13.7% of steps and
    7 zones 65.7%.
    """
    return max(1, math.ceil(num_rooms / max_rooms_per_ahu))


def assign_rooms_to_ahus(
    num_rooms: int,
    num_ahus: int,
    seed: int | None = None,
    max_rooms_per_ahu: int | None = None,
) -> list[list[str]]:
    """Assign rooms to AHUs with randomized but fair contiguous splits.

    Returns zone IDs in 'zone_id_X' format to match simulator expectations.

    `max_rooms_per_ahu` caps how many zones any one handler may serve. Without
    it the split was fair only on average: `max_fraction = 2 / num_ahus` lets a
    single handler take twice its share, and the last handler took whatever
    remained with no check at all. Across the shipped set that produced
    anything from 2 to 36 zones on one handler, so air-side capacity per zone
    varied 6x between buildings with no parameter expressing the intent.
    """
    if seed is not None:
        random.seed(seed)
    if num_ahus <= 0:
        raise ValueError("num_ahus must be positive")
    if num_rooms < num_ahus:
        raise ValueError(f"Need at least {num_ahus} rooms for {num_ahus} AHUs")

    min_fraction = 1.0 / (2 * num_ahus)
    max_fraction = 2.0 / num_ahus
    min_per_ahu = max(1, int(num_rooms * min_fraction))
    max_per_ahu = min(num_rooms, int(num_rooms * max_fraction))

    if max_rooms_per_ahu is not None:
        if max_rooms_per_ahu <= 0:
            raise ValueError("max_rooms_per_ahu must be positive")
        if num_ahus * max_rooms_per_ahu < num_rooms:
            raise ValueError(
                f"{num_ahus} AHUs cannot serve {num_rooms} rooms at "
                f"{max_rooms_per_ahu} rooms each; need at least "
                f"{required_ahus(num_rooms, max_rooms_per_ahu)}"
            )
        max_per_ahu = min(max_per_ahu, max_rooms_per_ahu)
        min_per_ahu = min(min_per_ahu, max_per_ahu)

    splits = []
    remaining = num_rooms
    for i in range(num_ahus - 1):
        ahus_left = num_ahus - i
        rooms_needed_for_rest = (ahus_left - 1) * min_per_ahu
        max_for_this = min(max_per_ahu, remaining - rooms_needed_for_rest)
        # Bound this handler's share from both sides. The upper bound keeps it
        # under the cap; the lower bound keeps the handlers after it under the
        # cap too, since whatever this one declines they have to absorb. The
        # last handler takes the remainder and gets no say, so if the earlier
        # ones take too little it is the one that ends up over the cap.
        rooms_rest_can_hold = (ahus_left - 1) * max_per_ahu
        min_for_this = max(1, min(min_per_ahu, max_for_this),
                           remaining - rooms_rest_can_hold)
        max_for_this = max(min_for_this, max_for_this)
        size = random.randint(min_for_this, max_for_this)
        splits.append(size)
        remaining -= size
    splits.append(remaining)

    if max_rooms_per_ahu is not None and max(splits) > max_rooms_per_ahu:
        raise AssertionError(
            f"split {splits} exceeds the cap of {max_rooms_per_ahu}"
        )

    result = []
    start = 1
    for size in splits:
        room_ids = [f"room_{i}" for i in range(start, start + size)]
        result.append(room_ids)
        start += size
    return result


OBSERVATION_NORMALIZER_CONFIG = """
##########################
### OBSERVATIONS
##########################

zone_air_temperature_sensor_normalizer/set_observation_normalization_constants.field_id = 'zone_air_temperature_sensor'
zone_air_temperature_sensor_normalizer/set_observation_normalization_constants.sample_mean = 295.0
zone_air_temperature_sensor_normalizer/set_observation_normalization_constants.sample_variance = 25.0

supply_air_temperature_sensor_normalizer/set_observation_normalization_constants.field_id = 'supply_air_temperature_sensor'
supply_air_temperature_sensor_normalizer/set_observation_normalization_constants.sample_mean = 290.0
supply_air_temperature_sensor_normalizer/set_observation_normalization_constants.sample_variance = 25.0

supply_air_temperature_setpoint_normalizer/set_observation_normalization_constants.field_id = 'supply_air_temperature_setpoint'
supply_air_temperature_setpoint_normalizer/set_observation_normalization_constants.sample_mean = 290.0
supply_air_temperature_setpoint_normalizer/set_observation_normalization_constants.sample_variance = 25.0

outside_air_temperature_sensor_normalizer/set_observation_normalization_constants.field_id = 'outside_air_temperature_sensor'
outside_air_temperature_sensor_normalizer/set_observation_normalization_constants.sample_mean = 295.0
outside_air_temperature_sensor_normalizer/set_observation_normalization_constants.sample_variance = 100.0

supply_water_temperature_setpoint_normalizer/set_observation_normalization_constants.field_id = 'supply_water_setpoint'
supply_water_temperature_setpoint_normalizer/set_observation_normalization_constants.sample_mean = 330.0
supply_water_temperature_setpoint_normalizer/set_observation_normalization_constants.sample_variance = 400.0

differential_pressure_setpoint_obs_normalizer/set_observation_normalization_constants.field_id = 'differential_pressure_setpoint'
differential_pressure_setpoint_obs_normalizer/set_observation_normalization_constants.sample_mean = 10.0
differential_pressure_setpoint_obs_normalizer/set_observation_normalization_constants.sample_variance = 36.0

run_status_normalizer/set_observation_normalization_constants.field_id = 'run_status'
run_status_normalizer/set_observation_normalization_constants.sample_mean = 0.5
run_status_normalizer/set_observation_normalization_constants.sample_variance = 0.25

supervisor_run_command_normalizer/set_observation_normalization_constants.field_id = 'supervisor_run_command'
supervisor_run_command_normalizer/set_observation_normalization_constants.sample_mean = 0.5
supervisor_run_command_normalizer/set_observation_normalization_constants.sample_variance = 0.25

request_count_observation_normalizer/set_observation_normalization_constants.field_id = 'request_count'
request_count_observation_normalizer/set_observation_normalization_constants.sample_mean = 50.0
request_count_observation_normalizer/set_observation_normalization_constants.sample_variance = 2500.0

# Shared by the VAV positions, damper and reheat valve. Without an entry a field
# falls through to mean 0 variance 1, which leaves these an order of magnitude
# smaller than the temperatures they sit beside.
#
# There is deliberately no entry for the air handlers' recirculation. These keys
# are matched against the measurement name exactly, and AirHandlerSystem
# presents every one of its children's fields under an ahu_N_ prefix, so no bare
# key here reaches an air handler -- which is why none of the AHU fields below
# are normalized either.
percentage_normalizer/set_observation_normalization_constants.field_id = 'percentage_command'
percentage_normalizer/set_observation_normalization_constants.sample_mean = 0.5
percentage_normalizer/set_observation_normalization_constants.sample_variance = 0.25

observation_normalizer_map = {
    'supply_air_damper_percentage_command': @percentage_normalizer/set_observation_normalization_constants(),
    'reheat_valve_percentage_command': @percentage_normalizer/set_observation_normalization_constants(),
    'zone_air_temperature_sensor': @zone_air_temperature_sensor_normalizer/set_observation_normalization_constants(),
    'supply_air_temperature_sensor': @supply_air_temperature_sensor_normalizer/set_observation_normalization_constants(),
    'supply_air_temperature_setpoint': @supply_air_temperature_setpoint_normalizer/set_observation_normalization_constants(),
    'supply_air_cooling_temperature_setpoint': @supply_air_temperature_setpoint_normalizer/set_observation_normalization_constants(),
    'supply_air_heating_temperature_setpoint': @supply_air_temperature_setpoint_normalizer/set_observation_normalization_constants(),
    'outside_air_temperature_sensor': @outside_air_temperature_sensor_normalizer/set_observation_normalization_constants(),
    'supply_water_setpoint': @supply_water_temperature_setpoint_normalizer/set_observation_normalization_constants(),
    'differential_pressure_setpoint': @differential_pressure_setpoint_obs_normalizer/set_observation_normalization_constants(),
    'run_status': @run_status_normalizer/set_observation_normalization_constants(),
    'supervisor_run_command': @supervisor_run_command_normalizer/set_observation_normalization_constants(),
    'cooling_request_count': @request_count_observation_normalizer/set_observation_normalization_constants(),
}

StandardScoreObservationNormalizer:
    normalization_constants = %observation_normalizer_map
"""


def generate_full_gin_config(
    floor_plan_path: str,
    zone_map_path: str,
    ahu_room_assignments: list[list[str]],
    config_name: str = "generated_config",
) -> str:
    """Generate a complete gin config file for the HVAC simulator."""
    num_ahus = len(ahu_room_assignments)

    room_lists = []
    for i, rooms in enumerate(ahu_room_assignments, 1):
        room_strs = ", ".join([f"'{r}'" for r in rooms])
        room_lists.append(f"ahu_{i}_rooms = [{room_strs}]")
    room_lists_str = "\n".join(room_lists)

    ahu_defs = []
    for i in range(1, num_ahus + 1):
        ahu_defs.append(
            f"""ahu_{i}/AirHandler:
  recirculation = %air_handler_recirculation_ratio
  heating_air_temp_setpoint = %air_handler_heating_setpoint
  cooling_air_temp_setpoint = %air_handler_cooling_setpoint
  fan_static_pressure = %fan_static_pressure
  fan_efficiency = %fan_efficiency
  max_air_flow_rate = 8.67
  sim_weather_controller = %weather_controller
  device_id = 'ahu_{i}'"""
        )
    ahu_defs_str = "\n\n".join(ahu_defs)

    ahu_mappings = ",\n    ".join(
        [f"@ahu_{i}/AirHandler(): %ahu_{i}_rooms" for i in range(1, num_ahus + 1)]
    )

    ahu_default_actions = []
    for i in range(1, num_ahus + 1):
        ahu_default_actions.extend(
            [
                f"'ahu_ahu_{i}_supply_air_temperature_setpoint': 293.0",
                f"'ahu_ahu_{i}_static_pressure_setpoint': 20000.0",
                f"'ahu_ahu_{i}_supervisor_run_command': 1.0",
            ]
        )
    ahu_default_actions_str = ",\n        ".join(ahu_default_actions)

    ahu_normalizers = []
    for i in range(1, num_ahus + 1):
        ahu_normalizers.extend(
            [
                f"'ahu_{i}_supply_air_temperature_setpoint': @supply_air_temperature_setpoint/set_action_normalization_constants()",
                f"'ahu_{i}_static_pressure_setpoint': @static_pressure_setpoint/set_action_normalization_constants()",
                f"'ahu_{i}_supervisor_run_command': @run_command/set_action_normalization_constants()",
            ]
        )
    ahu_normalizers_str = ",\n        ".join(ahu_normalizers)

    device_action_tuples = []
    for i in range(1, num_ahus + 1):
        device_action_tuples.extend(
            [
                f"('ahu', 'ahu_{i}_supervisor_run_command')",
                f"('ahu', 'ahu_{i}_supply_air_temperature_setpoint')",
                f"('ahu', 'ahu_{i}_static_pressure_setpoint')",
            ]
        )
    device_action_tuples.extend(
        [
            "('hws', 'supervisor_run_command')",
            "('hws', 'supply_water_setpoint')",
            "('hws', 'differential_pressure')",
        ]
    )
    device_action_tuples_str = ",\n        ".join(device_action_tuples)

    config = f"""###############################################################
# Generated HVAC Simulator Config: {config_name}
###############################################################

convection_coefficient = 100.0
ambient_high_temp = 285.0
ambient_low_temp = 280.0

sim/WeatherController:
  default_low_temp = %ambient_low_temp
  default_high_temp = %ambient_high_temp
  convection_coefficient = %convection_coefficient

weather_controller = @sim/WeatherController()

initial_temp = 294.0
control_volume_cm = 20
floor_height_cm = 300.0

floor_plan_filepath = "{floor_plan_path}"
zone_map_filepath = "{zone_map_path}"

# Material properties. Only density * heat_capacity reaches the solver, as a
# volumetric heat capacity in J/m3K; splitting it in two is for readability.
# The conductivities are effective values for a control volume, not handbook
# values for a bulk material.

# One control volume stands in for a whole facade, so its conductivity is an
# area-weighted blend of insulated opaque wall and glazing. Heat leaving the
# building crosses half that CV, then the surface film; at 0.21 W/mK the
# whole-building effective facade U measures 1.29 W/m2K and the envelope design
# load 12 W per m2 of floor across a 31 K difference. The mass is 20 cm of
# masonry over the opaque fraction and near nothing over the glazed fraction,
# smeared across the CV.
exterior_cv_conductivity = 0.21
exterior_cv_density = 2200.0
exterior_cv_heat_capacity = 910.0

# Interior partitions: gypsum board on a stud cavity, smeared over the CV.
# 0.5 W/mK puts a partition near 2.5 W/m2K, which is what keeps neighbouring
# zones thermally distinct.
interior_wall_cv_conductivity = 0.5
interior_wall_cv_density = 160.0
interior_wall_cv_heat_capacity = 900.0

# Occupied space: room air at its own density and heat capacity. The
# conductivity is deliberately far above still air's 0.03, because room air is
# mixed rather than stagnant and a high effective k is what holds a room close
# to isothermal. The furnishings are modelled separately, as the interior
# thermal mass node below, rather than being smeared into this density.
interior_cv_conductivity = 50.0
interior_cv_density = 1.2
interior_cv_heat_capacity = 1005.0

# Interior thermal mass: the floor slab and the furnishings, as a second node
# on each occupied CV, coupled to the room air across the CV's height. Only the
# top few centimetres of a slab take part in a daily cycle, so the effective
# volumetric capacity is far below concrete's: rho * c * z comes to 69 kJ per
# m2 of floor and K, on top of the 165 the partitions, facade and air already
# hold. k / z sets a 5 W/m2K coupling, which leaves the slab about four hours
# behind the room. Storage only, and deliberately: the node touches nothing but
# its own CV, so no value here can move the building's UA.
interior_mass_cv_conductivity = 15.0
interior_mass_cv_density = 23.0
interior_mass_cv_heat_capacity = 1000.0

inside_air_properties/MaterialProperties:
  conductivity = %interior_cv_conductivity
  heat_capacity = %interior_cv_heat_capacity
  density = %interior_cv_density

inside_wall_properties/MaterialProperties:
  conductivity = %interior_wall_cv_conductivity
  heat_capacity = %interior_wall_cv_heat_capacity
  density = %interior_wall_cv_density

building_exterior_properties/MaterialProperties:
  conductivity = %exterior_cv_conductivity
  heat_capacity = %exterior_cv_heat_capacity
  density = %exterior_cv_density

interior_mass_properties/MaterialProperties:
  conductivity = %interior_mass_cv_conductivity
  heat_capacity = %interior_mass_cv_heat_capacity
  density = %interior_mass_cv_density

sim/FloorPlanBasedBuilding:
  cv_size_cm = %control_volume_cm
  floor_height_cm = %floor_height_cm
  initial_temp = %initial_temp
  inside_air_properties = @inside_air_properties/MaterialProperties()
  inside_wall_properties = @inside_wall_properties/MaterialProperties()
  building_exterior_properties = @building_exterior_properties/MaterialProperties()
  interior_mass_properties = @interior_mass_properties/MaterialProperties()
  floor_plan_filepath = %floor_plan_filepath
  zone_map_filepath = %zone_map_filepath

morning_start_hour = 6
evening_start_hour = 19
heating_setpoint_day = 294
cooling_setpoint_day = 297
heating_setpoint_night = 289
cooling_setpoint_night = 298
time_zone = "US/Pacific"

hvac/SetpointSchedule:
  morning_start_hour = %morning_start_hour
  evening_start_hour = %evening_start_hour
  comfort_temp_window = (%heating_setpoint_day, %cooling_setpoint_day)
  eco_temp_window = (%heating_setpoint_night, %cooling_setpoint_night)
  time_zone = %time_zone

water_pump_differential_head = 6.0
water_pump_efficiency = 0.98
reheat_water_setpoint = 360.0
boiler_heating_rate = 0.5
boiler_cooling_rate = 0.1
fan_static_pressure = 1000.0
fan_efficiency = 0.9
air_handler_heating_setpoint = 285.0
air_handler_cooling_setpoint = 298.0
air_handler_recirculation_ratio = 0.3
vav_max_air_flowrate = 2.0
vav_reheat_water_flowrate = 0.108

{room_lists_str}

{ahu_defs_str}

hvac/AirHandlerSystem:
  ahus = {{
    {ahu_mappings}
  }}
  device_id = 'ahu'

hvac/WaterPump:
  water_pump_differential_head = %water_pump_differential_head
  water_pump_efficiency = %water_pump_efficiency

hvac/Boiler:
  reheat_water_setpoint = %reheat_water_setpoint
  heating_rate = %boiler_heating_rate
  cooling_rate = %boiler_cooling_rate

hvac/HotWaterSystem:
  pump = @hvac/WaterPump()
  boiler = @hvac/Boiler()
  device_id = 'hws'

sim/FloorPlanBasedHvac:
  air_handler = @hvac/AirHandlerSystem()
  hot_water_system = @hvac/HotWaterSystem()
  schedule = @hvac/SetpointSchedule()
  vav_max_air_flow_rate = %vav_max_air_flowrate
  vav_reheat_max_water_flow_factor = %vav_reheat_water_flowrate

time_step_sec = 300
internal_gain_w_per_m2 = 10.0
internal_gain_standby_fraction = 0.2
start_timestamp = '2023-07-10 06:00'

sim/to_timestamp.date_str = %start_timestamp

sim_building/SimulatorFlexibleGeometries:
  building = @sim/FloorPlanBasedBuilding()
  hvac = @sim/FloorPlanBasedHvac()
  weather_controller = %weather_controller
  time_step_sec = %time_step_sec
  internal_gain_w_per_m2 = %internal_gain_w_per_m2
  internal_gain_standby_fraction = %internal_gain_standby_fraction
  start_timestamp = @sim/to_timestamp()

work_occupancy = 2
nonwork_occupancy = 0.1

randomized_occupancy/RandomizedArrivalDepartureOccupancy:
  zone_assignment = %work_occupancy
  earliest_expected_arrival_hour = 6
  latest_expected_arrival_hour = 12
  earliest_expected_departure_hour = 13
  latest_expected_departure_hour = 19
  time_step_sec = %time_step_sec

SimulatorBuilding.simulator = @sim_building/SimulatorFlexibleGeometries()
SimulatorBuilding.occupancy = @randomized_occupancy/RandomizedArrivalDepartureOccupancy()

max_productivity_personhour_usd = 300.00
min_productivity_personhour_usd = 0.0
productivity_deviation_ref = 5.0
max_electricity_rate = {C_ELECTRICITY_PER_AHU * num_ahus}  # {C_ELECTRICITY_PER_AHU:,.0f} W/AHU x {num_ahus} AHUs
max_natural_gas_rate = {C_NATURAL_GAS_PER_AHU * num_ahus}  # {C_NATURAL_GAS_PER_AHU:,.0f} W/AHU x {num_ahus} AHUs
productivity_weight = 0.6
energy_cost_weight = 0.2
carbon_emission_weight = 0.2

SetpointEnergyCarbonRegretFunction.max_productivity_personhour_usd = %max_productivity_personhour_usd
SetpointEnergyCarbonRegretFunction.min_productivity_personhour_usd = %min_productivity_personhour_usd
SetpointEnergyCarbonRegretFunction.max_electricity_rate = %max_electricity_rate
SetpointEnergyCarbonRegretFunction.max_natural_gas_rate = %max_natural_gas_rate
SetpointEnergyCarbonRegretFunction.productivity_deviation_ref = %productivity_deviation_ref
SetpointEnergyCarbonRegretFunction.electricity_energy_cost = @ElectricityEnergyCost()
SetpointEnergyCarbonRegretFunction.natural_gas_energy_cost = @NaturalGasEnergyCost()
SetpointEnergyCarbonRegretFunction.productivity_weight = %productivity_weight
SetpointEnergyCarbonRegretFunction.energy_cost_weight = %energy_cost_weight
SetpointEnergyCarbonRegretFunction.carbon_emission_weight = %carbon_emission_weight

supply_water_bounded_action_normalizer/set_action_normalization_constants.min_normalized_value = -1.
supply_water_bounded_action_normalizer/set_action_normalization_constants.max_normalized_value = 1.0
supply_water_bounded_action_normalizer/set_action_normalization_constants.min_native_value = 310
supply_water_bounded_action_normalizer/set_action_normalization_constants.max_native_value = 350.0

supply_air_temperature_setpoint/set_action_normalization_constants.min_normalized_value = -1.
supply_air_temperature_setpoint/set_action_normalization_constants.max_normalized_value = 1.
supply_air_temperature_setpoint/set_action_normalization_constants.min_native_value = 285
supply_air_temperature_setpoint/set_action_normalization_constants.max_native_value = 305.0

differential_pressure_setpoint/set_action_normalization_constants.min_normalized_value = -1.
differential_pressure_setpoint/set_action_normalization_constants.max_normalized_value = 1.
differential_pressure_setpoint/set_action_normalization_constants.min_native_value = 0
differential_pressure_setpoint/set_action_normalization_constants.max_native_value = 20.0

static_pressure_setpoint/set_action_normalization_constants.min_normalized_value = -1.
static_pressure_setpoint/set_action_normalization_constants.max_normalized_value = 1.
static_pressure_setpoint/set_action_normalization_constants.min_native_value = 0
static_pressure_setpoint/set_action_normalization_constants.max_native_value = 20000.0

run_command/set_action_normalization_constants.min_normalized_value = -1.
run_command/set_action_normalization_constants.max_normalized_value = 1.
run_command/set_action_normalization_constants.min_native_value = 0.0
run_command/set_action_normalization_constants.max_native_value = 1.0

action_normalizer_map = {{
        'supply_water_setpoint': @supply_water_bounded_action_normalizer/set_action_normalization_constants(),
        'differential_pressure': @differential_pressure_setpoint/set_action_normalization_constants(),
        {ahu_normalizers_str},
        'supervisor_run_command': @run_command/set_action_normalization_constants(),
    }}

ActionConfig:
    action_normalizers = %action_normalizer_map

default_actions = {{
        'hws_supply_water_setpoint': 340.0,
        'hws_differential_pressure': 20.0,
        {ahu_default_actions_str},
        'hws_supervisor_run_command': 1.0,
    }}
{OBSERVATION_NORMALIZER_CONFIG}

discount_factor = 0.9
num_days_in_episode = 30
metrics_reporting_interval = 10
label = '{config_name}'
num_hod_features = 12
num_dow_features = 12

Environment.building = @SimulatorBuilding()
Environment.reward_function = @SetpointEnergyCarbonRegretFunction()
Environment.observation_normalizer = @StandardScoreObservationNormalizer()
Environment.action_config = @ActionConfig()
Environment.metrics_reporting_interval = %metrics_reporting_interval
Environment.discount_factor = %discount_factor
Environment.label = %label
Environment.num_days_in_episode = %num_days_in_episode
Environment.default_actions = %default_actions
Environment.num_hod_features = %num_hod_features
Environment.num_dow_features = %num_dow_features

HybridActionEnvironment.building = @SimulatorBuilding()
HybridActionEnvironment.reward_function = @SetpointEnergyCarbonRegretFunction()
HybridActionEnvironment.observation_normalizer = @StandardScoreObservationNormalizer()
HybridActionEnvironment.action_config = @ActionConfig()
HybridActionEnvironment.metrics_reporting_interval = %metrics_reporting_interval
HybridActionEnvironment.discount_factor = %discount_factor
HybridActionEnvironment.label = %label
HybridActionEnvironment.num_days_in_episode = %num_days_in_episode
HybridActionEnvironment.default_actions = %default_actions
HybridActionEnvironment.num_hod_features = %num_hod_features
HybridActionEnvironment.num_dow_features = %num_dow_features
HybridActionEnvironment.device_action_tuples = [
        {device_action_tuples_str},
    ]
"""
    return config


def save_simulation_config(
    bsp: BSPFloorPlan,
    output_dir: str,
    num_ahus: int = 2,
    config_name: str = "generated",
    seed: int | None = None,
) -> dict:
    """Generate and save all files needed for simulation.

    Args:
        bsp: BSPFloorPlan instance (must have called generate() first)
        output_dir: Directory to save output files
        num_ahus: Number of Air Handling Units
        config_name: Base name for output files
        seed: Random seed for AHU assignment

    Returns:
        Dict with paths and metadata
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    floor_plan_path = os.path.join(output_dir, f"{config_name}_floor_plan.npy")
    zone_map_path = os.path.join(output_dir, f"{config_name}_zone_map.npy")
    gin_path = os.path.join(output_dir, f"{config_name}.gin")

    bsp.save_for_simulator(floor_plan_path, zone_map_path)
    num_rooms = bsp.get_num_rooms()
    assignments = assign_rooms_to_ahus(num_rooms, num_ahus, seed)

    gin_config = generate_full_gin_config(
        floor_plan_path, zone_map_path, assignments, config_name
    )
    with open(gin_path, "w") as f:
        f.write(gin_config)

    return {
        "floor_plan": floor_plan_path,
        "zone_map": zone_map_path,
        "gin_config": gin_path,
        "num_rooms": num_rooms,
        "num_ahus": num_ahus,
        "assignments": assignments,
    }


def display_floor_plan(
    grid: np.ndarray, title: str = "Floor Plan", figsize: tuple[int, int] = (10, 10)
):
    """Display a floor plan grid.

    Grid encoding: 0=floor (white), 1=wall (dark), 2=door (blue)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=figsize)
    # 0=floor (white), 1=wall (dark gray), 2=door (blue)
    cmap = ListedColormap(["#FFFFFF", "#2C3E50", "#3498DB"])
    ax.imshow(grid, cmap=cmap, interpolation="nearest", vmin=0, vmax=2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def display_multiple(
    grids: list[np.ndarray],
    titles: list[str],
    cols: int = 3,
    figsize: tuple[int, int] = (15, 5),
):
    """Display multiple floor plans in a grid."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    rows = (len(grids) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1] * rows))
    axes = np.atleast_2d(axes).flatten()
    cmap = ListedColormap(["#FFFFFF", "#2C3E50", "#8B4513"])
    for idx, (grid, title) in enumerate(zip(grids, titles)):
        axes[idx].imshow(grid, cmap=cmap, interpolation="nearest", vmin=0, vmax=2)
        axes[idx].set_title(title, fontsize=11, fontweight="bold")
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    for idx in range(len(grids), len(axes)):
        axes[idx].axis("off")
    plt.tight_layout()
    plt.show()


def visualize_ahu_zones(
    bsp: BSPFloorPlan,
    assignments: list[list[str]],
    title: str = "AHU Zone Assignment",
    padding: int = 5,
    save_path: str | None = None,
    show: bool = True,
):
    """Visualize which rooms belong to which AHU with different colors."""
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    zone_map = bsp.export_zone_map(padding=padding)
    binary = (zone_map == 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary, connectivity=4)

    room_to_ahu = {}
    for ahu_idx, rooms in enumerate(assignments):
        for room in rooms:
            room_num = int(room.split("_")[-1])
            room_to_ahu[room_num] = ahu_idx

    num_ahus = len(assignments)
    base_colors = plt.cm.Set2(np.linspace(0, 1, max(num_ahus, 3)))[:num_ahus]
    # Light, distinct fills so black wall lines remain readable
    ahu_colors = [tuple(c[:3] * 0.55 + 0.45) for c in base_colors]

    h, w = zone_map.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)

    for room_num, ahu_idx in room_to_ahu.items():
        rgb[labels == room_num] = ahu_colors[ahu_idx]

    exterior = zone_map == 2
    rgb[exterior] = [1.0, 1.0, 1.0]

    wall_mask = zone_map == 1
    if np.any(wall_mask):
        wall_kernel = np.ones((2, 2), np.uint8)
        wall_display = cv2.dilate(
            wall_mask.astype(np.uint8), wall_kernel, iterations=1
        ).astype(bool)
        rgb[wall_display] = [0.0, 0.0, 0.0]

    # Upscale for crisp grid-aligned walls on screen
    scale = max(1, min(3, 900 // max(h, w)))
    if scale > 1:
        rgb = cv2.resize(
            rgb,
            (w * scale, h * scale),
            interpolation=cv2.INTER_NEAREST,
        )

    fig_size = max(8, min(14, max(h, w) * scale / 80))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=120)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    legend_elements = [
        Patch(facecolor=ahu_colors[i], edgecolor="black", label=f"AHU {i+1} ({len(assignments[i])} rooms)")
        for i in range(num_ahus)
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return rgb
