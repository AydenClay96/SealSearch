import cv2
import numpy as np


MIN_GEOMETRIC_INLIERS = 6
RANSAC_RANDOM_SEED = 1337


def create_coat_mask(image_bgr, background_threshold: int = 8):
    """Mask dark background connected to the image border, preserving dark coat pixels."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dark_pixels = (gray <= background_threshold).astype(np.uint8)
    component_count, labels = cv2.connectedComponents(dark_pixels, connectivity=8)
    if component_count <= 1:
        return np.full(gray.shape, 255, dtype=np.uint8)

    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    border_labels = border_labels[border_labels != 0]
    if border_labels.size == 0:
        return np.full(gray.shape, 255, dtype=np.uint8)

    background = np.isin(labels, border_labels).astype(np.uint8) * 255
    background = cv2.dilate(
        background, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    return cv2.bitwise_not(background)


def filter_geometric_matches(
    keypoints1, keypoints2, matches, reprojection_threshold: float = 5.0
):
    """Keep descriptor matches consistent with one RANSAC-estimated homography."""
    if len(matches) < MIN_GEOMETRIC_INLIERS:
        return []

    points1 = np.float32(
        [keypoints1[match.queryIdx].pt for match in matches]
    ).reshape(-1, 1, 2)
    points2 = np.float32(
        [keypoints2[match.trainIdx].pt for match in matches]
    ).reshape(-1, 1, 2)

    cv2.setRNGSeed(RANSAC_RANDOM_SEED)
    _, inlier_mask = cv2.findHomography(
        points1, points2, cv2.RANSAC, reprojection_threshold
    )
    if inlier_mask is None:
        return []

    inliers = [
        match
        for match, is_inlier in zip(matches, inlier_mask.ravel())
        if is_inlier
    ]
    return inliers if len(inliers) >= MIN_GEOMETRIC_INLIERS else []
