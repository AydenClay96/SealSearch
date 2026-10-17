"""Compare unmasked and coat-masked SIFT matching on held-out reference images."""

import argparse
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import cv2

from matching import create_coat_mask, filter_geometric_matches


def image_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_by_image_group(root: Path, test_fraction: float, seed: int):
    """Hold out exact-image groups while keeping each seal represented in training."""
    groups_by_seal = defaultdict(dict)
    for seal_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for image_path in sorted(seal_dir.glob("*.[jJ][pP][gG]")):
            groups_by_seal[seal_dir.name].setdefault(
                image_digest(image_path), []
            ).append(image_path)

    if not groups_by_seal:
        raise ValueError(f"No seal image folders found in {root}")

    rng = random.Random(seed)
    held_out_digests = set()
    for seal, groups in groups_by_seal.items():
        digests = list(groups)
        if len(digests) < 2:
            raise ValueError(
                f"{seal} needs at least two distinct images for held-out evaluation"
            )
        rng.shuffle(digests)
        count = min(len(digests) - 1, max(1, round(len(digests) * test_fraction)))
        held_out_digests.update(digests[:count])

    training = {}
    queries = []
    for seal, groups in groups_by_seal.items():
        references = [
            paths[0]
            for digest, paths in groups.items()
            if digest not in held_out_digests
        ]
        if not references:
            raise ValueError(f"Held-out split leaves {seal} without references")
        training[seal] = references
        queries.extend(
            (seal, paths[0])
            for digest, paths in groups.items()
            if digest in held_out_digests
        )

    return training, queries


def extract_features(image, detector, use_coat_mask: bool):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = create_coat_mask(image) if use_coat_mask else None
    return detector.detectAndCompute(gray, mask)


def geometric_score(query_features, reference_features) -> int:
    query_keypoints, query_descriptors = query_features
    reference_keypoints, reference_descriptors = reference_features
    if query_descriptors is None or reference_descriptors is None:
        return 0

    neighbors = cv2.BFMatcher().knnMatch(
        query_descriptors, reference_descriptors, k=2
    )
    good_matches = [
        pair[0]
        for pair in neighbors
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    return len(
        filter_geometric_matches(
            query_keypoints, reference_keypoints, good_matches
        )
    )


def evaluate(training, queries, extra_query=None, extra_label=None):
    detector = cv2.SIFT_create()
    for use_coat_mask in (False, True):
        feature_cache = {}
        paths = {
            path for references in training.values() for path in references
        } | {path for _, path in queries}
        for path in sorted(paths):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not read image: {path}")
            feature_cache[path] = extract_features(
                image, detector, use_coat_mask
            )

        if extra_query is not None:
            image = cv2.imread(str(extra_query))
            if image is None:
                raise ValueError(f"Could not read query image: {extra_query}")
            feature_cache[extra_query] = extract_features(
                image, detector, use_coat_mask
            )

        results = []
        for seal, query_path in queries:
            scores = {
                candidate: max(
                    geometric_score(
                        feature_cache[query_path], feature_cache[reference]
                    )
                    for reference in references
                )
                for candidate, references in training.items()
            }
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            results.append(
                (
                    seal,
                    ranked[0][0] == seal,
                    any(candidate == seal for candidate, _ in ranked[:3]),
                    ranked[0][1],
                    ranked[0][0],
                )
            )

        correct_top1 = sum(result[1] for result in results)
        correct_top3 = sum(result[2] for result in results)
        accepted = [result for result in results if result[3] >= 15]
        correct_accepted = sum(result[0] == result[4] for result in accepted)
        method = "coat-masked" if use_coat_mask else "unmasked"
        print(
            f"{method}: top-1 {correct_top1}/{len(results)}, "
            f"top-3 {correct_top3}/{len(results)}, "
            f">=15-inlier coverage {len(accepted)}/{len(results)}, "
            f"accepted correct {correct_accepted}/{len(accepted)}"
        )
        for seal in sorted(training):
            seal_results = [result for result in results if result[0] == seal]
            correct = sum(result[1] for result in seal_results)
            print(f"  {seal}: top-1 {correct}/{len(seal_results)}")
        if extra_query is not None:
            scores = {
                candidate: max(
                    geometric_score(
                        feature_cache[extra_query], feature_cache[reference]
                    )
                    for reference in references
                )
                for candidate, references in training.items()
            }
            predicted, score = sorted(
                scores.items(), key=lambda item: (-item[1], item[0])
            )[0]
            print(
                f"  extra query {extra_query}: predicted {predicted} "
                f"with {score} inliers; expected {extra_label}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("reference_seals"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--extra-query", type=Path)
    parser.add_argument("--extra-label")
    args = parser.parse_args()

    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1")
    if (args.extra_query is None) != (args.extra_label is None):
        parser.error("--extra-query and --extra-label must be used together")

    try:
        training, queries = split_by_image_group(
            args.dataset, args.test_fraction, args.seed
        )
        print(
            f"Dataset: {args.dataset}; identities: {len(training)}; "
            f"held-out image groups: {len(queries)}; seed: {args.seed}"
        )
        evaluate(training, queries, args.extra_query, args.extra_label)
    except (OSError, ValueError, cv2.error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
