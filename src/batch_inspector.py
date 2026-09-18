import os
import glob
import cv2
import pandas as pd
from ultralytics import YOLO

from image_quality import check_image_quality
from morphology import calculate_morphology
from zone_analysis import calculate_zone


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    "runs/detect/results/"
    "inspectai_defect_detector_v2-2/"
    "weights/best.pt"
)

IMAGE_FOLDER = "data/dataset/images/val"

OUTPUT_DIR = "results/batch"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "batch_inspection_results.csv"
)

CONFIDENCE_THRESHOLD = 0.15
IMAGE_SIZE = 320

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("INSPECTAI - BATCH INSPECTION")
print("=" * 70)

print()
print("Loading YOLO model...")

if not os.path.exists(MODEL_PATH):

    print("ERROR: Model not found.")
    print(f"Expected: {MODEL_PATH}")
    exit()

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# FIND IMAGES
# ============================================================

patterns = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp"
]

images = []

for pattern in patterns:

    images.extend(
        glob.glob(
            os.path.join(
                IMAGE_FOLDER,
                pattern
            )
        )
    )


images = sorted(images)


if len(images) == 0:

    print()
    print("ERROR: No images found.")
    print(
        f"Expected images inside: "
        f"{IMAGE_FOLDER}"
    )
    exit()


print()
print(
    f"Images found: {len(images)}"
)


# ============================================================
# BATCH RESULTS
# ============================================================

batch_results = []


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for image_number, image_path in enumerate(
    images,
    start=1
):

    filename = os.path.basename(
        image_path
    )

    print()
    print(
        f"[{image_number}/{len(images)}] "
        f"Processing: {filename}"
    )


    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:

        print("  Could not read image.")

        batch_results.append({

            "image": filename,

            "quality_status": "REJECT",

            "defect_count": 0,

            "affected_area_percentage": 0,

            "average_confidence": 0,

            "severity_score": 0,

            "severity": "UNKNOWN",

            "decision": "REJECT",

            "main_defect": "Unreadable Image"

        })

        continue


    height, width = image.shape[:2]

    image_area = width * height


    # --------------------------------------------------------
    # IMAGE QUALITY
    # --------------------------------------------------------

    quality = check_image_quality(
        image
    )


    print(
        f"  Quality: "
        f"{quality['quality_status']}"
    )


    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model.predict(

        source=image_path,

        conf=CONFIDENCE_THRESHOLD,

        imgsz=IMAGE_SIZE,

        verbose=False
    )


    result = results[0]


    detections = []


    # --------------------------------------------------------
    # EXTRACT DETECTIONS
    # --------------------------------------------------------

    if result.boxes is not None:

        boxes = result.boxes


        for i in range(
            len(boxes)
        ):

            # Bounding box
            box = boxes.xyxy[
                i
            ].cpu().numpy()

            x1, y1, x2, y2 = box


            # Confidence
            confidence = float(
                boxes.conf[
                    i
                ].cpu().numpy()
            )


            # Class
            class_id = int(
                boxes.cls[
                    i
                ].cpu().numpy()
            )

            class_name = model.names[
                class_id
            ]


            # ------------------------------------------------
            # BOUNDING BOX FEATURES
            # ------------------------------------------------

            bbox_width = max(
                0,
                x2 - x1
            )

            bbox_height = max(
                0,
                y2 - y1
            )

            bbox_area = (
                bbox_width *
                bbox_height
            )

            area_percentage = (
                bbox_area /
                image_area
            ) * 100


            # ------------------------------------------------
            # MORPHOLOGY
            # ------------------------------------------------

            morphology = (
                calculate_morphology(
                    image,
                    [
                        x1,
                        y1,
                        x2,
                        y2
                    ]
                )
            )


            # ------------------------------------------------
            # ZONE
            # ------------------------------------------------

            zone = calculate_zone(

                [
                    x1,
                    y1,
                    x2,
                    y2
                ],

                width,

                height
            )


            detections.append({

                "defect_type":
                    class_name,

                "confidence":
                    confidence,

                "area_percentage":
                    area_percentage,

                "morphology_area":
                    morphology[
                        "area_pixels"
                    ],

                "aspect_ratio":
                    morphology[
                        "aspect_ratio"
                    ],

                "perimeter":
                    morphology[
                        "perimeter_pixels"
                    ],

                "circularity":
                    morphology[
                        "circularity"
                    ],

                "zone":
                    zone[
                        "zone"
                    ]
            })


    # ========================================================
    # CALCULATE IMAGE-LEVEL STATISTICS
    # ========================================================

    defect_count = len(
        detections
    )


    if defect_count == 0:

        affected_area = 0

        largest_defect = 0

        average_confidence = 0

        main_defect = "None"

        severity_score = 0

        severity = "LOW"

        decision = "PASS"

        explanation = (
            "No defects detected."
        )


    else:

        # ----------------------------------------------------
        # Area
        # ----------------------------------------------------

        affected_area = sum(

            d["area_percentage"]

            for d in detections

        )

        affected_area = min(
            affected_area,
            100
        )


        # ----------------------------------------------------
        # Largest defect
        # ----------------------------------------------------

        largest_defect = max(

            d["area_percentage"]

            for d in detections

        )


        # ----------------------------------------------------
        # Average confidence
        # ----------------------------------------------------

        average_confidence = sum(

            d["confidence"]

            for d in detections

        ) / defect_count


        # ----------------------------------------------------
        # Main defect
        # ----------------------------------------------------

        defect_types = [

            d["defect_type"]

            for d in detections

        ]

        main_defect = max(

            set(defect_types),

            key=defect_types.count

        )


        # ====================================================
        # PROJECT-DEFINED SEVERITY SCORE
        # ====================================================

        # Area score
        if affected_area <= 2:
            area_score = 5

        elif affected_area <= 5:
            area_score = 12

        elif affected_area <= 10:
            area_score = 20

        elif affected_area <= 20:
            area_score = 26

        else:
            area_score = 30


        # Size score
        if largest_defect <= 2:
            size_score = 4

        elif largest_defect <= 5:
            size_score = 8

        elif largest_defect <= 10:
            size_score = 13

        elif largest_defect <= 20:
            size_score = 17

        else:
            size_score = 20


        # Count score
        if defect_count == 1:
            count_score = 4

        elif defect_count <= 3:
            count_score = 8

        elif defect_count <= 5:
            count_score = 12

        else:
            count_score = 15


        # Location score
        location_score = 0

        for d in detections:

            if d["zone"] == "Center":

                location_score += 10

            elif d["zone"] in [

                "Top-Center",
                "Middle-Left",
                "Middle-Right",
                "Bottom-Center"

            ]:

                location_score += 6

            else:

                location_score += 3


        location_score = min(
            location_score,
            20
        )


        # Defect type score
        high_risk = {

            "crazing",
            "pitted_surface",
            "rolled-in_scale"

        }


        if any(

            d["defect_type"]
            in high_risk

            for d in detections

        ):

            type_score = 15

        else:

            type_score = 7


        # ----------------------------------------------------
        # TOTAL SCORE
        # ----------------------------------------------------

        severity_score = (

            area_score
            + size_score
            + count_score
            + location_score
            + type_score

        )

        severity_score = min(
            severity_score,
            100
        )


        # ----------------------------------------------------
        # SEVERITY
        # ----------------------------------------------------

        if severity_score <= 30:

            severity = "LOW"

        elif severity_score <= 60:

            severity = "MEDIUM"

        else:

            severity = "HIGH"


        # ----------------------------------------------------
        # DECISION
        # ----------------------------------------------------

        if severity == "LOW":

            decision = "PASS"

        elif severity == "MEDIUM":

            decision = "INSPECT"

        else:

            decision = "REJECT"


        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        reasons = []


        if affected_area > 10:

            reasons.append(
                "Large affected area"
            )

        elif affected_area > 5:

            reasons.append(
                "Moderate affected area"
            )


        if largest_defect > 10:

            reasons.append(
                "Large individual defect"
            )


        if defect_count >= 5:

            reasons.append(
                "Multiple defects detected"
            )


        high_risk_detected = [

            d["defect_type"]

            for d in detections

            if d["defect_type"]
            in high_risk

        ]


        if high_risk_detected:

            reasons.append(
                "Higher-risk defect type detected"
            )


        if any(

            d["zone"] == "Center"

            for d in detections

        ):

            reasons.append(
                "Defect detected in center region"
            )


        low_confidence = sum(

            1

            for d in detections

            if d["confidence"] < 0.40

        )


        if low_confidence > 0:

            reasons.append(
                f"{low_confidence} "
                "low-confidence detection(s)"
            )


        if not reasons:

            reasons.append(
                "Defect characteristics "
                "remain within lower "
                "severity thresholds"
            )


        explanation = " | ".join(
            reasons
        )


    # ========================================================
    # QUALITY + DECISION
    # ========================================================

    # Poor image quality does not mean a physical defect.
    # It indicates that the inspection may require review.

    if quality["quality_status"] == "REVIEW":

        if decision == "PASS":

            decision = "INSPECT"

        explanation = (
            explanation
            + " | Image quality requires review"
        )


    # ========================================================
    # STORE RESULT
    # ========================================================

    batch_results.append({

        "image": filename,

        "width": width,

        "height": height,

        "quality_status":
            quality[
                "quality_status"
            ],

        "brightness":
            quality[
                "brightness"
            ],

        "contrast":
            quality[
                "contrast"
            ],

        "blur_score":
            quality[
                "blur_score"
            ],

        "defect_count":
            defect_count,

        "affected_area_percentage":
            round(
                affected_area,
                2
            ),

        "largest_defect_percentage":
            round(
                largest_defect,
                2
            ),

        "average_confidence":
            round(
                average_confidence,
                3
            ),

        "main_defect":
            main_defect,

        "severity_score":
            round(
                severity_score,
                2
            ),

        "severity":
            severity,

        "decision":
            decision,

        "explanation":
            explanation
    })


    print(
        f"  Defects: {defect_count} | "
        f"Severity: {severity} | "
        f"Decision: {decision}"
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    batch_results
)


# ============================================================
# SAVE CSV
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("BATCH INSPECTION COMPLETE")
print("=" * 70)

print(
    f"Images processed: "
    f"{len(results_df)}"
)

print()

print(
    "PASS:",
    len(
        results_df[
            results_df["decision"]
            == "PASS"
        ]
    )
)

print(
    "INSPECT:",
    len(
        results_df[
            results_df["decision"]
            == "INSPECT"
        ]
    )
)

print(
    "REJECT:",
    len(
        results_df[
            results_df["decision"]
            == "REJECT"
        ]
    )
)

print()

print(
    f"Results saved to:"
)

print(
    OUTPUT_FILE
)

print()
print("=" * 70)