import os
import csv
import cv2
import numpy as np


REFERENCE_DIR = "data/reference"
POSITIVE_DIR = "data/positive"
NEGATIVE_DIR = "data/negative"
RESULTS_DIR = "results"

MIN_GOOD_MATCHES = 40
MIN_INLIERS = 20
RATIO_TEST_THRESHOLD = 0.75


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def get_image_files(folder_path):
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    files = []

    if not os.path.exists(folder_path):
        return files

    for file_name in sorted(os.listdir(folder_path)):
        if file_name.lower().endswith(valid_ext):
            files.append(os.path.join(folder_path, file_name))

    return files


def load_grayscale_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image


def load_color_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image


def create_sift():
    return cv2.SIFT_create()


def compute_keypoints_and_descriptors(image_gray, sift):
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)
    return keypoints, descriptors


def match_descriptors(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < RATIO_TEST_THRESHOLD * n.distance:
                good_matches.append(m)

    return raw_matches, good_matches


def draw_raw_matches(ref_img, ref_kp, test_img, test_kp, good_matches, save_path, max_matches=50):
    matches_to_draw = sorted(good_matches, key=lambda x: x.distance)[:max_matches]

    match_img = cv2.drawMatches(
        ref_img,
        ref_kp,
        test_img,
        test_kp,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(save_path, match_img)


def compute_homography_and_box(ref_img, ref_kp, test_img, test_kp, good_matches, boxed_save_path, inliers_save_path):
    if len(good_matches) < 4:
        boxed_img = test_img.copy()
        cv2.imwrite(boxed_save_path, boxed_img)
        cv2.imwrite(inliers_save_path, boxed_img)
        return 0, None, None

    src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([test_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if homography is None or mask is None:
        boxed_img = test_img.copy()
        cv2.imwrite(boxed_save_path, boxed_img)
        cv2.imwrite(inliers_save_path, boxed_img)
        return 0, None, None

    inlier_count = int(mask.sum())

    h, w = ref_img.shape[:2]
    corners = np.float32([
        [0, 0],
        [0, h - 1],
        [w - 1, h - 1],
        [w - 1, 0]
    ]).reshape(-1, 1, 2)

    projected_corners = cv2.perspectiveTransform(corners, homography)

    boxed_img = test_img.copy()
    boxed_img = cv2.polylines(
        boxed_img,
        [np.int32(projected_corners)],
        isClosed=True,
        color=(0, 255, 0),
        thickness=4
    )
    cv2.imwrite(boxed_save_path, boxed_img)

    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i][0] == 1]

    inlier_img = cv2.drawMatches(
        ref_img,
        ref_kp,
        test_img,
        test_kp,
        inlier_matches[:80],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(inliers_save_path, inlier_img)

    return inlier_count, homography, projected_corners


def detection_decision(good_matches_count, inlier_count):
    return good_matches_count >= MIN_GOOD_MATCHES and inlier_count >= MIN_INLIERS


def analyze_single_image(reference_path, test_path, label):
    sift = create_sift()

    ref_gray = load_grayscale_image(reference_path)
    test_gray = load_grayscale_image(test_path)

    ref_color = load_color_image(reference_path)
    test_color = load_color_image(test_path)

    ref_kp, ref_desc = compute_keypoints_and_descriptors(ref_gray, sift)
    test_kp, test_desc = compute_keypoints_and_descriptors(test_gray, sift)

    if ref_desc is None or test_desc is None:
        return {
            "image_name": os.path.basename(test_path),
            "class": label,
            "reference_name": os.path.basename(reference_path),
            "reference_keypoints": len(ref_kp),
            "test_keypoints": len(test_kp),
            "good_matches": 0,
            "inliers": 0,
            "detected": False
        }

    _, good_matches = match_descriptors(ref_desc, test_desc)

    image_base = os.path.splitext(os.path.basename(test_path))[0]
    raw_path = os.path.join(RESULTS_DIR, f"{label}_{image_base}_matches.jpg")
    boxed_path = os.path.join(RESULTS_DIR, f"{label}_{image_base}_boxed.jpg")
    inliers_path = os.path.join(RESULTS_DIR, f"{label}_{image_base}_inliers.jpg")

    draw_raw_matches(ref_color, ref_kp, test_color, test_kp, good_matches, raw_path)

    inlier_count, _, _ = compute_homography_and_box(
        ref_color,
        ref_kp,
        test_color,
        test_kp,
        good_matches,
        boxed_path,
        inliers_path
    )

    detected = detection_decision(len(good_matches), inlier_count)

    return {
        "image_name": os.path.basename(test_path),
        "class": label,
        "reference_name": os.path.basename(reference_path),
        "reference_keypoints": len(ref_kp),
        "test_keypoints": len(test_kp),
        "good_matches": len(good_matches),
        "inliers": inlier_count,
        "detected": detected
    }


def save_results_csv(results, csv_path):
    fieldnames = [
        "image_name",
        "class",
        "reference_name",
        "reference_keypoints",
        "test_keypoints",
        "good_matches",
        "inliers",
        "detected"
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results):
    positive_results = [r for r in results if r["class"] == "positive"]
    negative_results = [r for r in results if r["class"] == "negative"]

    tp = sum(1 for r in positive_results if r["detected"])
    fn = sum(1 for r in positive_results if not r["detected"])
    fp = sum(1 for r in negative_results if r["detected"])
    tn = sum(1 for r in negative_results if not r["detected"])

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("\n----- DATASET SUMMARY -----")
    print(f"Positive images detected: {tp} / {len(positive_results)}")
    print(f"Negative images falsely detected: {fp} / {len(negative_results)}")
    print(f"True Positives  (TP): {tp}")
    print(f"False Negatives (FN): {fn}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives  (TN): {tn}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")


def main():
    ensure_results_dir()

    reference_images = get_image_files(REFERENCE_DIR)
    positive_images = get_image_files(POSITIVE_DIR)
    negative_images = get_image_files(NEGATIVE_DIR)

    print("Checking dataset folders...\n")
    print(f"Reference images found: {len(reference_images)}")
    print(f"Positive images found: {len(positive_images)}")
    print(f"Negative images found: {len(negative_images)}")

    if len(reference_images) == 0:
        raise ValueError("No reference images found in data/reference")

    reference_path = reference_images[0]
    print(f"\nUsing reference image: {os.path.basename(reference_path)}")

    results = []

    print("\n----- PROCESSING POSITIVE IMAGES -----")
    for img_path in positive_images:
        result = analyze_single_image(reference_path, img_path, "positive")
        results.append(result)
        print(
            f"{result['image_name']} | "
            f"good_matches={result['good_matches']} | "
            f"inliers={result['inliers']} | "
            f"detected={result['detected']}"
        )

    print("\n----- PROCESSING NEGATIVE IMAGES -----")
    for img_path in negative_images:
        result = analyze_single_image(reference_path, img_path, "negative")
        results.append(result)
        print(
            f"{result['image_name']} | "
            f"good_matches={result['good_matches']} | "
            f"inliers={result['inliers']} | "
            f"detected={result['detected']}"
        )

    csv_path = os.path.join(RESULTS_DIR, "detection_results.csv")
    save_results_csv(results, csv_path)
    print(f"\nResults saved to: {csv_path}")

    print_summary(results)