import cv2
import numpy as np


def calculate_morphology(image, bbox):
    """
    Calculate geometric properties of a detected defect.

    bbox format:
    [x1, y1, x2, y2]
    """

    x1, y1, x2, y2 = map(int, bbox)

    height, width = image.shape[:2]

    # Keep coordinates inside image
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))

    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:

        return {
            "area_pixels": 0,
            "aspect_ratio": 0,
            "perimeter_pixels": 0,
            "circularity": 0
        }

    # Crop detected region
    roi = image[
        y1:y2,
        x1:x2
    ]

    # Convert to grayscale
    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    # Threshold
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove small noise
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    # Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:

        defect_area = 0
        perimeter = 0
        circularity = 0

    else:

        # Largest contour
        contour = max(
            contours,
            key=cv2.contourArea
        )

        defect_area = float(
            cv2.contourArea(contour)
        )

        perimeter = float(
            cv2.arcLength(
                contour,
                True
            )
        )

        if perimeter > 0:

            circularity = (
                4 * np.pi * defect_area
            ) / (
                perimeter ** 2
            )

        else:

            circularity = 0


    # Bounding-box dimensions
    defect_width = x2 - x1
    defect_height = y2 - y1


    # Aspect ratio
    if defect_height > 0:

        aspect_ratio = (
            defect_width /
            defect_height
        )

    else:

        aspect_ratio = 0


    return {

        "area_pixels": round(
            defect_area,
            2
        ),

        "aspect_ratio": round(
            aspect_ratio,
            3
        ),

        "perimeter_pixels": round(
            perimeter,
            2
        ),

        "circularity": round(
            min(circularity, 1.0),
            3
        )
    }