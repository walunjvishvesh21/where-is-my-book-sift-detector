We used external resources such as Google Gemini, Stack Overflow, and certain YouTube videos, along with the course lecture slides, to strengthen our conceptual understanding of SIFT, feature matching, homography, and the overall computer vision pipeline. These resources were used only to understand the concepts, clarify doubts, and learn how classical object detection methods behave under scale, rotation, perspective, clutter, and lighting variation. The coding for this project was done by us; we did not use AI tools to generate the project code. One of the most interesting parts of the project was working with SIFT itself, because it showed how a classical feature-based method can still perform strong object matching in real-world scenes. We were especially curious about using our own dataset instead of a ready-made dataset, and that became the most fun part of the project. We spent a good amount of time collecting photos under different lighting conditions, viewing angles, rotations, and cluttered table arrangements. It was particularly interesting to test how the system would behave when three notebooks with somewhat similar building-style covers were present, since that made the task more challenging and realistic. One of the more difficult parts to implement was the homography-based detection stage, especially deciding when the matched points were geometrically consistent enough to confirm a real detection and draw the final bounding box. We encountered confusion and trial-and-error in that part, but after referring to Stack Overflow discussions and explanatory videos, we got a clearer idea of the logic and were able to complete the implementation ourselves.

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