import os
import xml.etree.ElementTree as ET
import shutil
import random
from pathlib import Path

# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "NEU-DET"
IMAGE_DIR = RAW_DIR / "IMAGES"
ANNOTATION_DIR = RAW_DIR / "ANNOTATIONS"

OUTPUT_DIR = BASE_DIR / "data" / "dataset"

TRAIN_IMAGES = OUTPUT_DIR / "images" / "train"
VAL_IMAGES = OUTPUT_DIR / "images" / "val"

TRAIN_LABELS = OUTPUT_DIR / "labels" / "train"
VAL_LABELS = OUTPUT_DIR / "labels" / "val"


# =========================
# DEFECT CLASSES
# =========================

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

CLASS_MAP = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# =========================
# CREATE DIRECTORIES
# =========================

for folder in [
    TRAIN_IMAGES,
    VAL_IMAGES,
    TRAIN_LABELS,
    VAL_LABELS
]:
    folder.mkdir(parents=True, exist_ok=True)


# =========================
# FIND IMAGES
# =========================

image_files = []

for extension in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
    image_files.extend(IMAGE_DIR.glob(extension))

image_files = sorted(image_files)

print(f"Images found: {len(image_files)}")


if len(image_files) == 0:
    raise FileNotFoundError(
        f"No images found in: {IMAGE_DIR}"
    )


# =========================
# TRAIN / VALIDATION SPLIT
# =========================

random.seed(42)

random.shuffle(image_files)

split_index = int(len(image_files) * 0.8)

train_images = image_files[:split_index]
val_images = image_files[split_index:]

print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")


# =========================
# XML → YOLO CONVERSION
# =========================

def convert_annotation(xml_file, image_file, label_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")

    image_width = int(size.find("width").text)
    image_height = int(size.find("height").text)

    labels = []

    for obj in root.findall("object"):

        class_name = obj.find("name").text.strip()

        if class_name not in CLASS_MAP:
            print(f"Warning: unknown class {class_name}")
            continue

        class_id = CLASS_MAP[class_name]

        bbox = obj.find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Clamp coordinates
        xmin = max(0, min(xmin, image_width))
        xmax = max(0, min(xmax, image_width))

        ymin = max(0, min(ymin, image_height))
        ymax = max(0, min(ymax, image_height))

        # Convert Pascal VOC → YOLO
        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height

        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        labels.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    with open(label_file, "w") as file:
        file.write("\n".join(labels))


# =========================
# PROCESS DATASET
# =========================

def process_split(images, image_output_dir, label_output_dir):

    converted = 0
    missing_annotations = 0

    for image_file in images:

        xml_file = ANNOTATION_DIR / f"{image_file.stem}.xml"

        if not xml_file.exists():
            print(f"Missing annotation: {xml_file.name}")
            missing_annotations += 1
            continue

        destination_image = image_output_dir / image_file.name
        destination_label = label_output_dir / f"{image_file.stem}.txt"

        shutil.copy2(
            image_file,
            destination_image
        )

        convert_annotation(
            xml_file,
            image_file,
            destination_label
        )

        converted += 1

    return converted, missing_annotations


# =========================
# CONVERT TRAINING DATA
# =========================

print("\nConverting training dataset...")

train_converted, train_missing = process_split(
    train_images,
    TRAIN_IMAGES,
    TRAIN_LABELS
)


# =========================
# CONVERT VALIDATION DATA
# =========================

print("\nConverting validation dataset...")

val_converted, val_missing = process_split(
    val_images,
    VAL_IMAGES,
    VAL_LABELS
)


# =========================
# CREATE data.yaml
# =========================

yaml_file = OUTPUT_DIR / "data.yaml"

yaml_content = f"""path: {OUTPUT_DIR.as_posix()}

train: images/train
val: images/val

names:
"""

for index, name in enumerate(CLASS_NAMES):
    yaml_content += f"  {index}: {name}\n"

with open(yaml_file, "w") as file:
    file.write(yaml_content)


# =========================
# FINAL REPORT
# =========================

print("\n===================================")
print("NEU-DET CONVERSION COMPLETE")
print("===================================")

print(f"Total images:       {len(image_files)}")
print(f"Training images:    {train_converted}")
print(f"Validation images:  {val_converted}")
print(f"Missing annotations: {train_missing + val_missing}")

print("\nDataset created at:")
print(OUTPUT_DIR)

print("\ndata.yaml created at:")
print(yaml_file)

print("\nClasses:")

for index, name in enumerate(CLASS_NAMES):
    print(f"{index}: {name}")

print("\n===================================")