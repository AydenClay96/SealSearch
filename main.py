"""Seal Identification Application
0. Delineate training and test images. DONE!
1. Load in and analyze/parameterize identified images.
2. Allow user to upload image.
3. Analyze/parameterize image (look for skin pattern).
4. Compare user image to identified images.
5. Return most alike (maybe a sorted list? Maybe based on tolerance?)
"""

import splitfolders
import os
import cv2
from pathlib import Path
import numpy as np
import time
from matching import create_coat_mask, filter_geometric_matches


def split_folders(folder: str, overwrite: bool = False) -> None:
    """Splits the image folders into training, test and evaluation datasets."""
    if not os.path.isdir("output") or overwrite:
        splitfolders.ratio(folder, output="output", seed=1337, ratio=(.8,.1,.1), group_prefix=None, group=None, formats=None, move=False, shuffle=True)

def remove_background(image):
    """Remove the background for classification."""
    height, width = image.shape[:2]
    margin = 0.05  # 5% margin
    x = int(width * margin)
    y = int(height * margin)
    rect = (x, y, width - 2*x, height - 2*y)
    mask = np.zeros((height, width))
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2) | (mask==0), 0, 1).astype("uint8")

    output_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    output_rgba[:, :, 3] = mask2 * 255
    return output_rgba


def extract_features(image_bgr, use_coat_mask: bool = False):
    """Convert images to grayscale and extract SIFT features."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    mask = create_coat_mask(image_bgr) if use_coat_mask else None
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    return keypoints, descriptors, gray

def match_patterns(keypoints1, desc1, keypoints2, desc2):
    """Score descriptor matches that agree on a geometric transformation."""
    if desc1 is None or desc2 is None:
        return 0.0, []

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for neighbors in matches:
        if len(neighbors) < 2:
            continue
        m, n = neighbors
        if m.distance < 0.75*n.distance:
            good_matches.append(m)

    # good_matches = filter_geometric_matches(keypoints1, keypoints2, good_matches)
    score = len(good_matches)
    return score, good_matches

def find_matches(
    test_img, folder: str = "./reference_seals", use_coat_mask: bool = False
):
    query_kp, query_desc, _ = extract_features(test_img, use_coat_mask)
    best_score = -1.0
    for seal_folder in Path(folder).iterdir():
            print(f"Working on {seal_folder}...")
            if seal_folder.is_dir():
                id = seal_folder.name
    
            for ref_path in seal_folder.glob("*.[jJ][pP][gG]"):
                ref_img = cv2.imread(str(ref_path))
                if ref_img is None:
                    continue
    
                ref_kp, ref_desc, _ = extract_features(ref_img, use_coat_mask)
    
                score, matches = match_patterns(
                    query_kp, query_desc, ref_kp, ref_desc
                )
    
                if score > best_score:
                    best_score = score
                    best_name = id
                    best_img = ref_img
                    best_match = matches
                    best_kp = ref_kp
    return best_score, best_name, best_img

def main():
    """ Main function """
    # print("Sorting the image folder!")
    # split_folders("refer/", True)

    print("Loaded user inputted image.")
    test_img = cv2.imread('cptnul_cptnul.jpg')

    best_score, best_name, best_img = find_matches(test_img, "./reference_seals")

    if best_score > 60.0:
        print(f"Success! Found a reasonable match!")        
    else:
        print(f"Failure! No reasonable match found!")

    print(f"Identified seal as: {best_name}")
    print(f"Geometric inlier count: {best_score}")
    test_img_norm = cv2.resize(test_img, (500, 500))
    best_img_norm = cv2.resize(best_img, (500, 500))
    cat_img = np.concatenate((test_img_norm, best_img_norm), axis=1)
    cv2.imshow("Image", cat_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
