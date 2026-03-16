import cv2
import os


REFERENCE_DIR = "data/reference"
POSITIVE_DIR = "data/positive"
NEGATIVE_DIR = "data/negative"
RESULTS_DIR = "results"


def load_grayscale_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def load_color_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def get_image_paths(folder_path):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = []

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(valid_extensions):
            image_files.append(os.path.join(folder_path, file_name))

    image_files.sort()
    return image_files


def get_good_matches(reference_gray, scene_gray):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(reference_gray, None)
    kp2, des2 = sift.detectAndCompute(scene_gray, None)

    if des1 is None or des2 is None:
        return kp1, kp2, [], [], des1, des2

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    return kp1, kp2, matches, good_matches, des1, des2


def save_match_image(ref_color, kp1, scene_color, kp2, good_matches, output_path):
    match_image = cv2.drawMatches(
        ref_color,
        kp1,
        scene_color,
        kp2,
        good_matches[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(output_path, match_image)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Checking dataset folders...\n")

    reference_images = get_image_paths(REFERENCE_DIR)
    positive_images = get_image_paths(POSITIVE_DIR)
    negative_images = get_image_paths(NEGATIVE_DIR)

    print(f"Reference images found: {len(reference_images)}")
    print(f"Positive images found: {len(positive_images)}")
    print(f"Negative images found: {len(negative_images)}\n")

    if len(reference_images) == 0 or len(positive_images) == 0 or len(negative_images) == 0:
        print("Need at least one reference, one positive, and one negative image.")
        return

    reference_image_path = reference_images[0]
    positive_image_path = positive_images[0]
    negative_image_path = negative_images[0]

    reference_gray = load_grayscale_image(reference_image_path)
    positive_gray = load_grayscale_image(positive_image_path)
    negative_gray = load_grayscale_image(negative_image_path)

    reference_color = load_color_image(reference_image_path)
    positive_color = load_color_image(positive_image_path)
    negative_color = load_color_image(negative_image_path)

    print("Images loaded successfully.\n")

    print("----- REFERENCE vs POSITIVE -----")
    kp1, kp2, matches_pos, good_pos, des1, des2 = get_good_matches(reference_gray, positive_gray)

    print(f"Reference keypoints: {len(kp1)}")
    print(f"Positive keypoints: {len(kp2)}")
    if des1 is not None and des2 is not None:
        print(f"Reference descriptor shape: {des1.shape}")
        print(f"Positive descriptor shape: {des2.shape}")
    print(f"Total raw matches: {len(matches_pos)}")
    print(f"Good matches after ratio test: {len(good_pos)}\n")

    save_match_image(
        reference_color,
        kp1,
        positive_color,
        kp2,
        good_pos,
        os.path.join(RESULTS_DIR, "reference_vs_positive_matches.jpg")
    )

    print("----- REFERENCE vs NEGATIVE -----")
    kp1n, kp2n, matches_neg, good_neg, des1n, des2n = get_good_matches(reference_gray, negative_gray)

    print(f"Reference keypoints: {len(kp1n)}")
    print(f"Negative keypoints: {len(kp2n)}")
    if des1n is not None and des2n is not None:
        print(f"Reference descriptor shape: {des1n.shape}")
        print(f"Negative descriptor shape: {des2n.shape}")
    print(f"Total raw matches: {len(matches_neg)}")
    print(f"Good matches after ratio test: {len(good_neg)}\n")

    save_match_image(
        reference_color,
        kp1n,
        negative_color,
        kp2n,
        good_neg,
        os.path.join(RESULTS_DIR, "reference_vs_negative_matches.jpg")
    )

    print("----- FINAL COMPARISON -----")
    print(f"Good matches with positive image: {len(good_pos)}")
    print(f"Good matches with negative image: {len(good_neg)}")

    if len(good_pos) > len(good_neg):
        print("Result: Positive image matches better than negative image.")
    else:
        print("Result: Negative image matched unexpectedly well. We may need stricter filtering.")


if __name__ == "__main__":
    main()