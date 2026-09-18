import os
import glob
import cv2
import pandas as pd
from ultralytics import YOLO
from zone_analysis import calculate_zone

# Import morphology analysis
from morphology import calculate_morphology


# ============================================================
# CONFIGURATION
# ============================================================

CONFIDENCE_THRESHOLD = 0.15
IMAGE_SIZE = 320

OUTPUT_DIR = "results/detections"
CSV_FILE = os.path.join(OUTPUT_DIR, "detection_results.csv")
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)


# ============================================================
# FIND TRAINED MODEL
# ============================================================

possible_models = [
    "runs/detect/results/inspectai_defect_detector_v2-2/weights/best.pt",
    "results/inspectai_defect_detector_v2-2/weights/best.pt",

    # Older model paths
    "results/inspectai_defect_detector_fast/weights/best.pt",
    "runs/detect/inspectai_defect_detector_fast/weights/best.pt",
    "runs/detect/results/inspectai_defect_detector_fast/weights/best.pt"
]

model_path = None

for path in possible_models:
    if os.path.exists(path):
        model_path = path
        break


# ============================================================
# SEARCH ENTIRE PROJECT IF MODEL NOT FOUND
# ============================================================

if model_path is None:

    found_models = glob.glob(
        "**/best.pt",
        recursive=True
    )

    if found_models:
        # Prefer the V2 model if available
        v2_models = [
            path for path in found_models
            if "inspectai_defect_detector_v2-2" in path
        ]

        if v2_models:
            model_path = v2_models[0]
        else:
            model_path = found_models[0]


# ============================================================
# CHECK MODEL
# ============================================================

if model_path is None:

    print("ERROR: best.pt was not found.")
    print()
    print("Please make sure your trained YOLO model exists.")
    exit()


print("=" * 60)
print("INSPECTAI DEFECT ANALYZER")
print("=" * 60)

print(f"Model: {model_path}")


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading YOLO model...")

model = YOLO(model_path)

print("Model loaded successfully.")


# ============================================================
# FIND VALIDATION IMAGE
# ============================================================

image_patterns = [
    "data/dataset/images/val/*.jpg",
    "data/dataset/images/val/*.png",
    "data/dataset/images/val/*.jpeg"
]

images = []

for pattern in image_patterns:
    images.extend(glob.glob(pattern))


# ============================================================
# CHECK IMAGES
# ============================================================

if len(images) == 0:

    print()
    print("ERROR: No validation images found.")
    print()
    print("Expected images inside:")
    print("data/dataset/images/val/")
    exit()


# Use first validation image
image_path = images[0]

print()
print(f"Image: {image_path}")


# ============================================================
# READ IMAGE
# ============================================================

image = cv2.imread(image_path)

if image is None:

    print()
    print("ERROR: Could not read image.")
    exit()


image_height, image_width = image.shape[:2]

image_area = image_width * image_height


print(
    f"Image size: "
    f"{image_width} x {image_height}"
)


# ============================================================
# RUN YOLO DETECTION
# ============================================================

print()
print("Running defect detection...")

results = model.predict(
    source=image_path,
    conf=CONFIDENCE_THRESHOLD,
    imgsz=IMAGE_SIZE,
    verbose=False
)


# ============================================================
# EXTRACT DETECTIONS
# ============================================================

detections = []

result = results[0]


if result.boxes is not None:

    boxes = result.boxes

    for i in range(len(boxes)):

        # ----------------------------------------------------
        # BOUNDING BOX
        # ----------------------------------------------------

        box = boxes.xyxy[i].cpu().numpy()

        x1, y1, x2, y2 = box


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence = float(
            boxes.conf[i].cpu().numpy()
        )


        # ----------------------------------------------------
        # CLASS
        # ----------------------------------------------------

        class_id = int(
            boxes.cls[i].cpu().numpy()
        )

        class_name = model.names[class_id]


        # ----------------------------------------------------
        # BOUNDING BOX DIMENSIONS
        # ----------------------------------------------------

        width = max(
            0,
            x2 - x1
        )

        height = max(
            0,
            y2 - y1
        )


        # ----------------------------------------------------
        # BOUNDING BOX AREA
        # ----------------------------------------------------

        bbox_area = width * height


        # ----------------------------------------------------
        # AREA PERCENTAGE
        # ----------------------------------------------------

        area_percentage = (
            bbox_area / image_area
        ) * 100


        # ====================================================
        # MORPHOLOGY ANALYSIS
        # ====================================================

        morphology = calculate_morphology(
            image,
            [x1, y1, x2, y2]
        )


        # ========================================================
        # SPATIAL / ZONE ANALYSIS
        # ========================================================

        zone = calculate_zone(
            [x1, y1, x2, y2],
            image_width,
            image_height
        )


        # ====================================================
        # STORE DETECTION
        # ====================================================

        detections.append({

            "image": os.path.basename(
                image_path
            ),

            "defect_type": class_name,

            "confidence": round(
                confidence,
                4
            ),

            # Bounding box coordinates
            "x1": round(
                float(x1),
                2
            ),

            "y1": round(
                float(y1),
                2
            ),

            "x2": round(
                float(x2),
                2
            ),

            "y2": round(
                float(y2),
                2
            ),

            # Bounding box dimensions
            "width": round(
                float(width),
                2
            ),

            "height": round(
                float(height),
                2
            ),

            # Bounding box area
            "bbox_area": round(
                float(bbox_area),
                2
            ),

            "area_percentage": round(
                float(area_percentage),
                4
            ),

            # =================================================
            # MORPHOLOGY FEATURES
            # =================================================

            "morphology_area": morphology[
                "area_pixels"
            ],

            "aspect_ratio": morphology[
                "aspect_ratio"
            ],

            "perimeter": morphology[
                "perimeter_pixels"
            ],

            "circularity": morphology[
                "circularity"
            ],

            # Spatial analysis
            "zone": zone["zone"],
            "zone_row": zone["zone_row"],
            "zone_column": zone["zone_column"],
            "center_x": zone["center_x"],
            "center_y": zone["center_y"],
            "relative_x": zone["relative_x"],
            "relative_y": zone["relative_y"]
        })


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 60)
print("DETECTION RESULTS")
print("=" * 60)


if len(detections) == 0:

    print()
    print("No defects detected.")

    print()
    print(
        "Try lowering the confidence threshold "
        "if necessary."
    )


else:

    print(
        f"Defects detected: "
        f"{len(detections)}"
    )

    print()

    for i, detection in enumerate(
        detections,
        start=1
    ):

        print(
            f"{i}. "
            f"{detection['defect_type']} | "
            f"Confidence: "
            f"{detection['confidence']:.2f} | "
            f"BBox Area: "
            f"{detection['area_percentage']:.2f}%"
        )

        print(
            f"   Width: "
            f"{detection['width']} px"
        )

        print(
            f"   Height: "
            f"{detection['height']} px"
        )

        print(
            f"   Morphology Area: "
            f"{detection['morphology_area']} px²"
        )

        print(
            f"   Aspect Ratio: "
            f"{detection['aspect_ratio']}"
        )

        print(
            f"   Perimeter: "
            f"{detection['perimeter']} px"
        )

        print(
            f"   Circularity: "
            f"{detection['circularity']}"
        )

        print()


# ============================================================
# SAVE CSV
# ============================================================

columns = [

    # Basic information
    "image",
    "defect_type",
    "confidence",

    # Bounding box
    "x1",
    "y1",
    "x2",
    "y2",

    # Dimensions
    "width",
    "height",

    # Area
    "bbox_area",
    "area_percentage",

    # Morphology
    "morphology_area",
    "aspect_ratio",
    "perimeter",
    "circularity",


    # Spatial analysis
    "zone",
    "zone_row",
    "zone_column",
    "center_x",
    "center_y",
    "relative_x",
    "relative_y"
]


df = pd.DataFrame(
    detections,
    columns=columns
)


df.to_csv(
    CSV_FILE,
    index=False
)


print(
    f"CSV saved to: "
    f"{CSV_FILE}"
)

print(
    f"Rows written: "
    f"{len(df)}"
)


# ============================================================
# SAVE ANNOTATED IMAGE
# ============================================================

annotated = result.plot()


annotated_path = os.path.join(
    ANNOTATED_DIR,
    "sample_detection.jpg"
)


cv2.imwrite(
    annotated_path,
    annotated
)


print(
    f"Annotated image saved to: "
    f"{annotated_path}"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)