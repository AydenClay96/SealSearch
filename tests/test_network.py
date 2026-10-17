import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from network import (
    SealDataset,
    SealEmbeddingNetwork,
    _class_similarity_scores,
    _evaluate_embeddings,
)


class SealNetworkTests(unittest.TestCase):
    def test_embedding_network_returns_unit_normalized_resnet_features(self):
        model = SealEmbeddingNetwork(pretrained=False).eval()

        with torch.inference_mode():
            embeddings = model(torch.rand((2, 3, 64, 64)))

        self.assertEqual(tuple(embeddings.shape), (2, 1024))
        self.assertTrue(torch.allclose(embeddings.norm(dim=1), torch.ones(2)))

    def test_dataset_labels_follow_sorted_class_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            split_dir = Path(temporary_directory) / "train"
            for pixel_value, class_name in enumerate(
                ("Seal_SealID_10", "Seal_SealID_3"), start=100
            ):
                class_dir = split_dir / class_name
                class_dir.mkdir(parents=True)
                cv2.imwrite(
                    str(class_dir / "sample.jpg"),
                    np.full((12, 20, 3), pixel_value, dtype=np.uint8),
                )

            dataset = SealDataset(temporary_directory, "train", image_size=32)
            image, label = dataset[0]

            self.assertEqual(dataset.class_names, ["Seal_SealID_10", "Seal_SealID_3"])
            self.assertEqual(label, 0)
            self.assertEqual(tuple(image.shape), (3, 32, 32))
            self.assertEqual(image.dtype, torch.uint8)

    def test_dataset_excludes_images_duplicated_in_an_earlier_split(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pixels = np.full((12, 20, 3), 127, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), pixels)
            image_bytes = image_path.read_bytes()
            image_path.unlink()

            train_dir = root / "train" / "Seal_SealID_3"
            val_dir = root / "val" / "Seal_SealID_3"
            train_dir.mkdir(parents=True)
            val_dir.mkdir(parents=True)
            (train_dir / "train.jpg").write_bytes(image_bytes)
            (val_dir / "duplicate.jpg").write_bytes(image_bytes)
            cv2.imwrite(
                str(val_dir / "unique.jpg"),
                np.full((12, 20, 3), 200, dtype=np.uint8),
            )

            train_data = SealDataset(root, "train", image_size=32)
            val_data = SealDataset(
                root,
                "val",
                class_names=train_data.class_names,
                image_size=32,
                excluded_hashes=train_data.image_hashes,
            )

            self.assertEqual(val_data.excluded_count, 1)
            self.assertEqual(len(val_data), 1)

    def test_nearest_reference_scores_and_metrics_rank_the_matching_class(self):
        query_embeddings = torch.tensor([[0.99, 0.01], [0.02, 0.98]])
        query_labels = torch.tensor([0, 1])
        reference_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
        reference_labels = torch.tensor([0, 1, 2])

        scores = _class_similarity_scores(
            query_embeddings, reference_embeddings, reference_labels, 3
        )
        metrics = _evaluate_embeddings(
            query_embeddings,
            query_labels,
            reference_embeddings,
            reference_labels,
            3,
            torch.device("cpu"),
        )

        self.assertEqual(scores.argmax(dim=1).tolist(), [0, 1])
        self.assertEqual(metrics.top1_accuracy, 1.0)
        self.assertEqual(metrics.top3_accuracy, 1.0)
        self.assertEqual(metrics.macro_accuracy, 1.0)
        self.assertEqual(metrics.num_images, 2)


if __name__ == "__main__":
    unittest.main()
