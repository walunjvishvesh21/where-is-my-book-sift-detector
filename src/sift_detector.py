#We used external resources such as Google Gemini, Stack Overflow, and certain YouTube videos,
#  along with the course lecture slides, to strengthen our conceptual understanding of SIFT,
#  feature matching, homography, and the overall computer vision pipeline. 
# These resources were used only to understand the concepts, clarify doubts, and learn
#  how classical object detection methods behave under scale, rotation, perspective, clutter,
#  and lighting variation. The coding for this project was done by us; we did not use AI tools 
# to generate the project code. One of the most interesting parts of the project was 
# working with SIFT itself, because it showed how a classical feature-based method can 
# still perform strong object matching in real-world scenes. We were especially curious 
# about using our own dataset instead of a ready-made dataset, and that became the most
#  fun part of the project. We spent a good amount of time collecting photos under different
#  lighting conditions, viewing angles, rotations, and cluttered table arrangements. 
# It was particularly interesting to test how the system would behave when three notebooks
#  with somewhat similar building-style covers were present, since that made the task more 
# challenging and realistic. One of the more difficult parts to implement was the 
# homography-based detection stage, especially deciding when the matched points were 
# geometrically consistent enough to confirm a real detection and draw the final bounding box.
#  We encountered confusion and trial-and-error in that part, but after referring to 
# Stack Overflow discussions and explanatory videos, we got a clearer idea of the logic and
#  were able to complete the implementation ourselves.


import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt



REFERENCE_DIR = "data/reference"
POSITIVE_DIR = "data/positive"
NEGATIVE_DIR = "data/negative"
RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "detection_results.csv")

RATIO_TEST_THRESHOLD = 0.75
MIN_GOOD_MATCHES = 150
MIN_INLIERS = 120

# Our first step here is we are creating a results folder if it does not already exist.
# This will store all of our results images, csv etc.
def ensure_results_folder():
    os.makedirs(RESULTS_DIR, exist_ok=True)

# Now this function will return sorted list of valid image file paths from folder
def get_image_paths(folder_path):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    files = []

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(valid_extensions):
            files.append(os.path.join(folder_path, file_name))

    return sorted(files)

# Now since SIFT works on intensity patterns and not colours and
# Feature detection is typically more stable and simpler in grayscale
# we loaded an image in Grayscale for our SIFT feature detection
def load_image_gray(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image

# Now this function below loads an image in normal colour
# Now grayscale we used was for detection but for visualization we use colour images
# SO we load image in colour for drawing results from visualization
def load_image_color(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image

# Now we wanted to find the key-points and descriptors
# This was the core step i believe because without descriptors there is nothing to match
# SO this function below detects keypoints and descriptors from a gray scale image
def detect_sift_features(gray_image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    return keypoints, descriptors


# Now we used match_descriptors to match descriptors between two images using:
# BF MATCHER (Brute force using one descriptor againts many descriptors in other images and finding the closest one)
#  and Lowe's Ratio test( for each descriptor it finds the best match and second best match)
# We used ratio test to remove ambiguous matches
# SO basically we matched descriptors using BF Matcher and only kept relaible matches using LOWE's ratio test.
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


# We created this function cause we all know raw matches alone cant prove object presence
# so what homography does is If the target book is rotated in the scene, homography can still map its corners properly.
# we  Estimated homography with RANSAC and to identify geometrically consistent inlier matches
def compute_homography(kp1, kp2, good_matches):
    if len(good_matches) < 4:
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask

# Now this function below just matches inliner between reference and test image.
# Now raw matches are quite noisy so inliner matches are more trustworthy.
# so we drew  saved only the geometrically consistent inlier matches between two images
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


# this function below uses homography to project the 4 corners of the reference image onto the test image 
# and draw a green box/polygon.
#  so If homography is correct, then the reference book’s corners should map to where the actual book is in the scene.
# this function projects the reference image corners into the test image and draws a detection box
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

# This was one of the main core functions.
# cause it compares one reference image with one test image and return match, homography, and detection results
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

# Now we needed to choose the best result with not just one but 5 reference images so
# so this function Compares a test image against all reference images and return the best matching result
def evaluate_against_all_references(reference_paths, test_path):
    all_results = []

    for ref_path in reference_paths:
        result = evaluate_reference_vs_test(ref_path, test_path)
        all_results.append(result)

    best_result = max(all_results, key=lambda x: (x["inliers"], x["good_matches"]))
    return best_result, all_results

# then we  Saved the  final image-level detection results to a CSV file
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


# we also created and saved a confusion matrix image using final classification counts
def save_confusion_matrix(tp, fn, fp, tn, save_path):
    matrix = [
        [tp, fn],
        [fp, tn]
    ]

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")

    plt.xticks([0, 1], ["Predicted Positive", "Predicted Negative"])
    plt.yticks([0, 1], ["Actual Positive", "Actual Negative"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=14)

    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# This is the overall controller of our entire project.
# It basically runs the full dataset evaluation pipeline and prints the  final detection performance metrics
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

    confusion_matrix_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    save_confusion_matrix(tp, fn, fp, tn, confusion_matrix_path)
    print(f"Confusion matrix saved to: {confusion_matrix_path}")