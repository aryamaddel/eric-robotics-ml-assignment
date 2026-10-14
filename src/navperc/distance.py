"""Monocular distance from a bounding box using the pinhole camera model.

For an object of known real height H (metres) standing roughly upright and facing the
camera, perspective projection gives its image height h (pixels) as

    h = f_y * H / Z   =>   Z = f_y * H / h

where f_y is the focal length in pixels (K[1, 1]) and Z is depth along the optical axis.
f_y = f_mm / pixel_pitch_mm, so this is the "focal length + pixel size" relation; when
only a field of view is known, f_y = (image_height / 2) / tan(vfov / 2).

Range to the object (what a robot usually wants) also accounts for its lateral offset:
the box centre (u, v) back-projects to the ray K^-1 [u, v, 1], so the 3D point is
Z * K^-1 [u, v, 1] and its Euclidean norm is the range.

Height is used rather than width because cones/signs are rotationally symmetric in
height but not in width, and barriers vary in length far more than in height.

A second estimator uses the ground plane: for an object resting on flat ground, the
bottom edge of the box at row v_b, a camera at height h_cam above the ground with zero
pitch, gives Z = f_y * h_cam / (v_b - c_y). It is independent of the object's size, which
helps for barriers whose height varies, but breaks on slopes and with camera pitch.
"""

from dataclasses import dataclass, field

import numpy as np

# Typical real-world heights (metres). Configurable; documented in the README.
DEFAULT_HEIGHTS = {
    "cone": 0.70,       # common 28" road cone (small 18" cones are 0.45 m)
    "barrier": 1.00,    # jersey / water-filled barriers are ~0.8-1.07 m
    "stop_sign": 0.75,  # 30" (0.75 m) octagon, the US/EU standard size; face only, not the post
}


@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    height_m: float | None = None  # camera height above ground, enables the ground-plane estimate

    @classmethod
    def from_K(cls, K, height_m=None):
        K = np.asarray(K, dtype=float)
        return cls(K[0, 0], K[1, 1], K[0, 2], K[1, 2], height_m)

    @classmethod
    def from_fov(cls, width, height, hfov_deg, height_m=None):
        """Approximate intrinsics from horizontal field of view, assuming square pixels."""
        f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
        return cls(f, f, width / 2, height / 2, height_m)

    def scaled(self, sx: float, sy: float) -> "Camera":
        """Intrinsics for an image resized by (sx, sy)."""
        return Camera(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, self.height_m)


@dataclass
class DistanceEstimator:
    camera: Camera
    heights: dict = field(default_factory=lambda: dict(DEFAULT_HEIGHTS))

    def depth_from_height(self, cls_name: str, box) -> float:
        x0, y0, x1, y1 = box
        h_px = max(y1 - y0, 1e-6)
        return float(self.camera.fy * self.heights[cls_name] / h_px)

    def depth_from_ground(self, box) -> float | None:
        if self.camera.height_m is None:
            return None
        dv = box[3] - self.camera.cy
        return float(self.camera.fy * self.camera.height_m / dv) if dv > 1 else None

    def range_from_depth(self, depth: float, box) -> float:
        u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        ray = np.array([(u - self.camera.cx) / self.camera.fx, (v - self.camera.cy) / self.camera.fy, 1.0])
        return float(depth * np.linalg.norm(ray))

    def estimate(self, cls_name: str, box, image_size=None) -> dict:
        """Return depth (along optical axis) and range (Euclidean) in metres.

        If the top of the box is cut off by the image border its pixel height is too small
        (distance overestimated), so we switch to the ground-plane estimate when the bottom
        contact point is still visible and the camera height is known.
        """
        z = self.depth_from_height(cls_name, box)
        method = "height"
        if image_size is not None:
            top_cut, bottom_cut = cut_edges(box, *image_size)
            zg = self.depth_from_ground(box) if top_cut and not bottom_cut else None
            if zg is not None:
                z, method = zg, "ground"
        return {"depth": z, "range": self.range_from_depth(z, box), "method": method}


def cut_edges(box, width, height, margin=2):
    """(top_cut, bottom_cut): whether the box touches the top / bottom image border."""
    return box[1] <= margin, box[3] >= height - margin
