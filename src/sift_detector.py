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


def get_image_paths(folder_path):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = []

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(valid_extensions):
            full_path = os.path.join(folder_path, file_name)
            image_files.append(full_path)

    image_files.sort()
    return image_files


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Checking dataset folders...\n")

    reference_images = get_image_paths(REFERENCE_DIR)
    positive_images = get_image_paths(POSITIVE_DIR)
    negative_images = get_image_paths(NEGATIVE_DIR)

    print(f"Reference images found: {len(reference_images)}")
    print(f"Positive images found: {len(positive_images)}")
    print(f"Negative images found: {len(negative_images)}\n")

    if len(reference_images) == 0 or len(positive_images) == 0:
        print("Need at least one reference image and one positive image.")
        return

    reference_image_path = reference_images[0]
    positive_image_path = positive_images[0]

    reference_image = load_grayscale_image(reference_image_path)
    positive_image = load_grayscale_image(positive_image_path)

    print("Reference and positive images loaded successfully.")
    print(f"Reference image shape: {reference_image.shape}")
    print(f"Positive image shape: {positive_image.shape}\n")

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(reference_image, None)
    kp2, des2 = sift.detectAndCompute(positive_image, None)

    print(f"Reference keypoints: {len(kp1)}")
    print(f"Positive keypoints: {len(kp2)}")

    if des1 is not None:
        print(f"Reference descriptor shape: {des1.shape}")
    else:
        print("Reference descriptors not found.")

    if des2 is not None:
        print(f"Positive descriptor shape: {des2.shape}")
    else:
        print("Positive descriptors not found.")


if __name__ == "__main__":
    main()