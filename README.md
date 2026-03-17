# Where's My Book? A SIFT-Based Object Finder for Cluttered Real-World Scenes

This project is a classical computer vision project that detects a target red **"Victoria Memorial"** notebook in cluttered indoor scenes using SIFT feature matching and homography verification.

## Project Goal
The goal is to determine whether the target notebook is present in a scene image and, if present, localize it using geometric feature matching.

## Dataset
The dataset was collected manually using a smartphone camera.

- 5 reference images
- 20 positive images
- 11 negative images
- 36 total images

### Dataset Categories
- `data/reference/` : clean views of the target notebook
- `data/positive/` : scene images containing the target notebook
- `data/negative/` : scene images not containing the target notebook

## Method
The detection pipeline uses:

1. grayscale image conversion  
2. SIFT keypoint and descriptor extraction  
3. brute-force descriptor matching using BFMatcher  
4. Lowe's ratio test  
5. homography estimation using RANSAC  
6. inlier-based verification for final detection  

## Multi-Reference Detection
Each test image is compared against all 5 reference images.  
The best reference is selected using:

- highest inlier count
- highest good match count

This makes the detector more robust to:

- scale variation
- perspective change
- rotation
- viewpoint change

## Final Detection Rule
A target is considered detected only if:

- a valid homography is found
- good matches >= 150
- inliers >= 120

## Challenges
A key challenge in this project was that some negative images contained notebooks with similar printed building structures, which produced false matches. To reduce these false positives, the final system used stricter thresholds on good matches and homography inliers.

## Final Results
Using the tuned multi-reference detector:

- True Positives (TP): 20
- False Negatives (FN): 0
- False Positives (FP): 3
- True Negatives (TN): 8

### Metrics
- Accuracy: 0.9032
- Precision: 0.8696
- Recall: 1.0000

The tuned multi-reference version reduced false positives while maintaining perfect recall on all positive images in the collected dataset.

## Output Files
The `results/` folder contains:

- detection visualizations
- boxed detections
- inlier match visualizations
- `detection_results.csv`
- `confusion_matrix.png`

## How to Run

### 1. Activate environment
```bash
conda activate bookdetect

python -m pip install -r requirements.txt
python main.py