import os
import cv2
import csv
import numpy as np


REFERENCE_DIR = "data/reference"
POSITIVE_DIR = "data/positive"
NEGATIVE_DIR = "data/negative"
RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "detection_results.csv")

RATIO_TEST_THRESHOLD = 0.75
MIN_GOOD_MATCHES = 150
MIN_INLIERS = 120


def ensure_results_folder():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def get_image_paths(folder_path):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    files = []

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(valid_extensions):
            files.append(os.path.join(folder_path, file_name))

    return sorted(files)


def load_image_gray(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image


def load_image_color(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image


def detect_sift_features(gray_image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    return keypoints, descriptors


def match_descriptors(desc1, desc2):
    bf = cv2.BFMatcher()
    knn_matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue

        m, n = pair
        if m.distance < RATIO_TEST_THRESHOLD * n.distance:
            good_matches.append(m)

    return good_matches


def compute_homography(kp1, kp2, good_matches):
    if len(good_matches) < 4:
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask


def draw_inlier_matches(ref_img, ref_kp, test_img, test_kp, good_matches, mask, save_path):
    if mask is None:
        return

    matches_mask = mask.ravel().tolist()

    result = cv2.drawMatches(
        ref_img,
        ref_kp,
        test_img,
        test_kp,
        good_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matches_mask,
        flags=2
    )

    cv2.imwrite(save_path, result)


def draw_detected_box(ref_gray, test_color, H, save_path):
    h, w = ref_gray.shape

    corners = np.float32([
        [0, 0],
        [0, h - 1],
        [w - 1, h - 1],
        [w - 1, 0]
    ]).reshape(-1, 1, 2)

    projected_corners = cv2.perspectiveTransform(corners, H)

    boxed_image = test_color.copy()
    cv2.polylines(
        boxed_image,
        [np.int32(projected_corners)],
        isClosed=True,
        color=(0, 255, 0),
        thickness=4
    )

    cv2.imwrite(save_path, boxed_image)


def evaluate_reference_vs_test(reference_path, test_path):
    ref_gray = load_image_gray(reference_path)
    test_gray = load_image_gray(test_path)
    test_color = load_image_color(test_path)

    ref_kp, ref_desc = detect_sift_features(ref_gray)
    test_kp, test_desc = detect_sift_features(test_gray)

    if ref_desc is None or test_desc is None:
        return {
            "reference_name": os.path.basename(reference_path),
            "test_name": os.path.basename(test_path),
            "good_matches": 0,
            "inliers": 0,
            "detected": False,
            "homography_found": False,
            "ref_keypoints": 0 if ref_kp is None else len(ref_kp),
            "test_keypoints": 0 if test_kp is None else len(test_kp),
            "box_image": None,
            "inlier_image": None
        }

    good_matches = match_descriptors(ref_desc, test_desc)
    H, mask = compute_homography(ref_kp, test_kp, good_matches)

    inliers = 0
    homography_found = H is not None and mask is not None

    if homography_found:
        inliers = int(mask.sum())

    detected = (
        homography_found
        and len(good_matches) >= MIN_GOOD_MATCHES
        and inliers >= MIN_INLIERS
    )

    test_base = os.path.splitext(os.path.basename(test_path))[0]
    ref_base = os.path.splitext(os.path.basename(reference_path))[0]

    inlier_image_path = os.path.join(
        RESULTS_DIR,
        f"{test_base}__bestmatch_with__{ref_base}__inliers.jpg"
    )
    box_image_path = os.path.join(
        RESULTS_DIR,
        f"{test_base}__bestmatch_with__{ref_base}__boxed.jpg"
    )

    if homography_found:
        draw_inlier_matches(
            load_image_color(reference_path),
            ref_kp,
            load_image_color(test_path),
            test_kp,
            good_matches,
            mask,
            inlier_image_path
        )
        draw_detected_box(ref_gray, test_color, H, box_image_path)
    else:
        inlier_image_path = None
        box_image_path = None

    return {
        "reference_name": os.path.basename(reference_path),
        "test_name": os.path.basename(test_path),
        "good_matches": len(good_matches),
        "inliers": inliers,
        "detected": detected,
        "homography_found": homography_found,
        "ref_keypoints": len(ref_kp),
        "test_keypoints": len(test_kp),
        "box_image": box_image_path,
        "inlier_image": inlier_image_path
    }


def evaluate_against_all_references(reference_paths, test_path):
    all_results = []

    for ref_path in reference_paths:
        result = evaluate_reference_vs_test(ref_path, test_path)
        all_results.append(result)

    best_result = max(all_results, key=lambda x: (x["inliers"], x["good_matches"]))
    return best_result, all_results


def save_results_to_csv(rows):
    fieldnames = [
        "image_name",
        "actual_class",
        "best_reference",
        "good_matches",
        "inliers",
        "detected"
    ]

    with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    ensure_results_folder()

    print("Checking dataset folders...\n")

    reference_paths = get_image_paths(REFERENCE_DIR)
    positive_paths = get_image_paths(POSITIVE_DIR)
    negative_paths = get_image_paths(NEGATIVE_DIR)

    print(f"Reference images found: {len(reference_paths)}")
    print(f"Positive images found: {len(positive_paths)}")
    print(f"Negative images found: {len(negative_paths)}\n")

    if len(reference_paths) == 0:
        print("No reference images found.")
        return

    results_rows = []

    tp = 0
    fn = 0
    fp = 0
    tn = 0

    print("----- PROCESSING POSITIVE IMAGES -----")
    for pos_path in positive_paths:
        best_result, _ = evaluate_against_all_references(reference_paths, pos_path)

        image_name = os.path.basename(pos_path)
        detected = best_result["detected"]

        print(
            f"{image_name} | best_ref={best_result['reference_name']} | "
            f"good_matches={best_result['good_matches']} | "
            f"inliers={best_result['inliers']} | detected={detected}"
        )

        results_rows.append({
            "image_name": image_name,
            "actual_class": "positive",
            "best_reference": best_result["reference_name"],
            "good_matches": best_result["good_matches"],
            "inliers": best_result["inliers"],
            "detected": detected
        })

        if detected:
            tp += 1
        else:
            fn += 1

    print("\n----- PROCESSING NEGATIVE IMAGES -----")
    for neg_path in negative_paths:
        best_result, _ = evaluate_against_all_references(reference_paths, neg_path)

        image_name = os.path.basename(neg_path)
        detected = best_result["detected"]

        print(
            f"{image_name} | best_ref={best_result['reference_name']} | "
            f"good_matches={best_result['good_matches']} | "
            f"inliers={best_result['inliers']} | detected={detected}"
        )

        results_rows.append({
            "image_name": image_name,
            "actual_class": "negative",
            "best_reference": best_result["reference_name"],
            "good_matches": best_result["good_matches"],
            "inliers": best_result["inliers"],
            "detected": detected
        })

        if detected:
            fp += 1
        else:
            tn += 1

    save_results_to_csv(results_rows)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\nResults saved to: {CSV_PATH}\n")

    print("----- DATASET SUMMARY -----")
    print(f"Positive images detected: {tp} / {tp + fn}")
    print(f"Negative images falsely detected: {fp} / {fp + tn}")
    print(f"True Positives  (TP): {tp}")
    print(f"False Negatives (FN): {fn}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives  (TN): {tn}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")