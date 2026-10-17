import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from evaluate_matching import image_digest, split_by_image_group
from matching import create_coat_mask, filter_geometric_matches


class FilterGeometricMatchesTests(unittest.TestCase):
    def test_held_out_split_keeps_exact_duplicate_images_together(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for seal in ("Seal_1", "Seal_2"):
                (root / seal).mkdir()

            first = np.full((20, 20, 3), 20, dtype=np.uint8)
            duplicate_path = root / "Seal_1" / "first.jpg"
            self.assertTrue(cv2.imwrite(str(duplicate_path), first))
            (root / "Seal_1" / "duplicate.jpg").write_bytes(
                duplicate_path.read_bytes()
            )
            cv2.imwrite(
                str(root / "Seal_1" / "second.jpg"),
                np.full((20, 20, 3), 80, dtype=np.uint8),
            )
            for index, value in enumerate((30, 100, 160)):
                cv2.imwrite(
                    str(root / "Seal_2" / f"{index}.jpg"),
                    np.full((20, 20, 3), value, dtype=np.uint8),
                )

            training, queries = split_by_image_group(root, 0.34, 1337)

            training_digests = {
                image_digest(path)
                for paths in training.values()
                for path in paths
            }
            query_digests = {image_digest(path) for _, path in queries}
            self.assertFalse(training_digests & query_digests)
            self.assertEqual(sum(label == "Seal_1" for label, _ in queries), 1)
            self.assertEqual(sum(label == "Seal_2" for label, _ in queries), 1)

    def test_coat_mask_excludes_border_background_and_keeps_dark_coat(self):
        image = cv2.cvtColor(
            np.full((40, 40), 80, dtype=np.uint8), cv2.COLOR_GRAY2BGR
        )
        image[:3] = 0
        image[-3:] = 0
        image[:, :3] = 0
        image[:, -3:] = 0
        image[18:22, 18:22] = 4

        mask = create_coat_mask(image)

        self.assertEqual(int(mask[0, 0]), 0)
        self.assertEqual(int(mask[20, 20]), 255)
        self.assertEqual(int(mask[10, 10]), 255)

    def test_keeps_scaled_rotated_matches_and_rejects_outliers(self):
        keypoints1 = []
        keypoints2 = []
        matches = []

        for y in (20, 80, 140):
            for x in (20, 80, 140):
                transformed_x = 1.4 * x - 0.3 * y + 45
                transformed_y = 0.3 * x + 1.4 * y + 25
                keypoints1.append(cv2.KeyPoint(float(x), float(y), 1))
                keypoints2.append(
                    cv2.KeyPoint(float(transformed_x), float(transformed_y), 1)
                )
                matches.append(cv2.DMatch(len(matches), len(matches), 0, 1.0))

        for x, y, outlier_x, outlier_y in (
            (10, 160, 200, 10),
            (70, 170, 5, 200),
            (150, 15, 180, 180),
        ):
            keypoints1.append(cv2.KeyPoint(float(x), float(y), 1))
            keypoints2.append(cv2.KeyPoint(float(outlier_x), float(outlier_y), 1))
            matches.append(cv2.DMatch(len(matches), len(matches), 0, 1.0))

        inliers = filter_geometric_matches(keypoints1, keypoints2, matches)

        self.assertEqual(len(inliers), 9)

    def test_rejects_fewer_than_minimum_geometric_matches(self):
        keypoints = [cv2.KeyPoint(float(i), float(i), 1) for i in range(5)]
        matches = [cv2.DMatch(i, i, 0, 1.0) for i in range(5)]

        self.assertEqual(filter_geometric_matches(keypoints, keypoints, matches), [])


if __name__ == "__main__":
    unittest.main()
