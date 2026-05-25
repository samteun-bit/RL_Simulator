import math
import numpy as np


class Track:
    """Rectangular oval track defined by outer and inner bounding boxes."""

    OUTER_W = 40.0  # half-width
    OUTER_H = 25.0  # half-height
    INNER_W = 25.0
    INNER_H = 12.0
    RAY_MAX_DIST = 40.0

    def __init__(self):
        self._wall_segments = self._build_segments()
        ow, oh = self.OUTER_W, self.OUTER_H
        iw, ih = self.INNER_W, self.INNER_H
        # diagonal of the playable area
        self._max_dist = math.sqrt((ow * 2) ** 2 + (oh * 2) ** 2)

    def _build_segments(self):
        ow, oh = self.OUTER_W, self.OUTER_H
        iw, ih = self.INNER_W, self.INNER_H

        outer = [
            ((-ow, -oh), (ow, -oh)),
            ((ow, -oh), (ow, oh)),
            ((ow, oh), (-ow, oh)),
            ((-ow, oh), (-ow, -oh)),
        ]
        inner = [
            ((-iw, -ih), (iw, -ih)),
            ((iw, -ih), (iw, ih)),
            ((iw, ih), (-iw, ih)),
            ((-iw, ih), (-iw, -ih)),
        ]
        return outer + inner

    def get_wall_segments(self):
        return self._wall_segments

    @staticmethod
    def _ray_segment_intersect(ox, oy, dx, dy, ax, ay, bx, by):
        sx, sy = bx - ax, by - ay
        denom = dx * sy - dy * sx
        if abs(denom) < 1e-10:
            return None
        t = ((ax - ox) * sy - (ay - oy) * sx) / denom
        s = ((ax - ox) * dy - (ay - oy) * dx) / denom
        if t > 1e-6 and 0.0 <= s <= 1.0:
            return t
        return None

    def cast_ray(self, ox, oy, angle_rad):
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        min_t = self.RAY_MAX_DIST
        for (ax, ay), (bx, by) in self._wall_segments:
            t = self._ray_segment_intersect(ox, oy, dx, dy, ax, ay, bx, by)
            if t is not None and t < min_t:
                min_t = t
        return min_t / self.RAY_MAX_DIST  # normalized [0, 1]

    def cast_ray_world(self, ox, oy, angle_rad):
        """Returns (hit_x, hit_y, normalized_dist) for visualization."""
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        min_t = self.RAY_MAX_DIST
        for (ax, ay), (bx, by) in self._wall_segments:
            t = self._ray_segment_intersect(ox, oy, dx, dy, ax, ay, bx, by)
            if t is not None and t < min_t:
                min_t = t
        return ox + dx * min_t, oy + dy * min_t, min_t / self.RAY_MAX_DIST

    def is_in_bounds(self, x, y):
        ow, oh = self.OUTER_W, self.OUTER_H
        iw, ih = self.INNER_W, self.INNER_H
        in_outer = -ow <= x <= ow and -oh <= y <= oh
        in_inner = -iw <= x <= iw and -ih <= y <= ih
        return in_outer and not in_inner

    def progress_along_track(self, x, y):
        """Counterclockwise progress [0, 1] around track centerline."""
        cx = (self.OUTER_W + self.INNER_W) / 2
        cy = (self.OUTER_H + self.INNER_H) / 2
        # Use atan2 relative to track center (0,0)
        # Scale so the oval maps more evenly to angle
        angle = math.atan2(y / cy, x / cx)
        return (angle + math.pi) / (2 * math.pi)  # [0, 1]

    def get_start_pose(self):
        start_y = -(self.OUTER_H + self.INNER_H) / 2
        return 0.0, start_y, 0.0  # x, y, heading_rad (0 = east)

    def get_start_finish_line(self):
        """Vertical line at x=0 crossing the bottom straight (start = finish)."""
        return (0.0, -self.OUTER_H), (0.0, -self.INNER_H)

    @property
    def max_distance(self):
        return self._max_dist
