import cv2
import numpy as np


def check_image_quality(image):
    """
    Analyze image quality before defect detection.

    Returns:
        dict containing:
        - width
        - height
        - brightness
        - contrast
        - blur_score
        - quality_status
        - quality_message
    """

    if image is None:
        return {
            "quality_status": "REJECT",
            "quality_message": "Image could not be read."
        }

    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Brightness
    brightness = float(np.mean(gray))

    # Contrast
    contrast = float(np.std(gray))

    # Blur detection
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    issues = []

    # Minimum resolution
    if width < 100 or height < 100:
        issues.append("Low image resolution")

    # Very dark image
    if brightness < 40:
        issues.append("Image is too dark")

    # Very bright image
    if brightness > 220:
        issues.append("Image is too bright")

    # Low contrast
    if contrast < 20:
        issues.append("Low contrast")

    # Blur threshold
    if blur_score < 50:
        issues.append("Image may be blurry")

    # Final quality decision
    if issues:
        quality_status = "REVIEW"
        quality_message = "; ".join(issues)
    else:
        quality_status = "PASS"
        quality_message = "Image quality is suitable for inspection."

    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur_score": round(blur_score, 2),
        "quality_status": quality_status,
        "quality_message": quality_message
    }