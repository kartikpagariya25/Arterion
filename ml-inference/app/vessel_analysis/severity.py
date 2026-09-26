import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt


def _skeleton_width_profile(vessel_mask: np.ndarray):
    skeleton = skeletonize(vessel_mask > 0)
    dist = distance_transform_edt(vessel_mask > 0)
    ys, xs = np.nonzero(skeleton)
    widths = dist[ys, xs] * 2
    return np.column_stack([ys, xs]), widths


def estimate_stenosis_percent(vessel_mask: np.ndarray, stenosis_bbox: tuple, window: float = 30.0) -> float:
    """stenosis_bbox: (x, y, w, h) in the same pixel space as vessel_mask."""
    points, widths = _skeleton_width_profile(vessel_mask)
    if len(points) == 0:
        return 0.0

    x, y, w, h = stenosis_bbox
    cx, cy = x + w / 2, y + h / 2

    dists_to_lesion = np.sqrt((points[:, 0] - cy) ** 2 + (points[:, 1] - cx) ** 2)

    lesion_radius = max(w, h) / 2
    at_lesion = dists_to_lesion <= lesion_radius
    nearby = (dists_to_lesion > lesion_radius) & (dists_to_lesion <= lesion_radius + window)

    if not at_lesion.any() or not nearby.any():
        return 0.0

    min_width_at_lesion = widths[at_lesion].min()
    reference_width = np.median(widths[nearby])

    if reference_width <= 0:
        return 0.0

    percent = (1 - min_width_at_lesion / reference_width) * 100
    return float(np.clip(percent, 0, 100))


def severity_band(percent: float) -> str:
    if percent < 50:
        return "mild"
    if percent <= 70:
        return "moderate"
    return "severe"
