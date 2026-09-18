import os
import glob
import sqlite3
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="InspectAI | Industrial Inspection",
    page_icon="I",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "results",
    "inspectai_defect_detector_v2-2",
    "weights",
    "best.pt"
)

BATCH_CSV = os.path.join(
    BASE_DIR,
    "results",
    "batch",
    "batch_inspection_results.csv"
)

HISTORY_DB = os.path.join(
    BASE_DIR,
    "results",
    "inspectai_history.db"
)

DETECTION_DIR = os.path.join(
    BASE_DIR,
    "results",
    "detections"
)

ANNOTATED_DIR = os.path.join(
    DETECTION_DIR,
    "annotated"
)

os.makedirs(DETECTION_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_DB), exist_ok=True)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

CONFIDENCE_THRESHOLD = 0.15
IMAGE_SIZE = 320


# ============================================================
# MODEL PERFORMANCE
# ============================================================

MODEL_METRICS = {
    "Precision": 62.0,
    "Recall": 71.7,
    "mAP@50": 72.4,
    "mAP@50-95": 40.9
}

CLASS_METRICS = pd.DataFrame(
    [
        ["crazing", 39.1, 24.3, 29.2, 10.3],
        ["inclusion", 61.6, 73.4, 73.0, 40.6],
        ["patches", 74.8, 96.0, 92.1, 60.7],
        ["pitted_surface", 77.6, 81.6, 86.6, 53.2],
        ["rolled_in_scale", 49.1, 59.9, 59.4, 28.7],
        ["scratches", 69.7, 95.1, 94.1, 52.1],
    ],
    columns=[
        "Defect",
        "Precision",
        "Recall",
        "mAP@50",
        "mAP@50-95"
    ]
)


# ============================================================
# DARK INDUSTRIAL UI
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background:
            radial-gradient(
                circle at 20% 0%,
                rgba(40, 70, 110, 0.20),
                transparent 35%
            ),
            #080b10;
        color: #f2f4f7;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #202833;
    }

    section[data-testid="stSidebar"] * {
        color: #e6e9ed;
    }

    /* Main content */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* Header */
    .main-title {
        font-size: 2.4rem;
        font-weight: 750;
        letter-spacing: -0.8px;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #8f9baa;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* KPI cards */
    .metric-card {
        background: linear-gradient(
            145deg,
            #111720,
            #0d1219
        );
        border: 1px solid #222b36;
        border-radius: 14px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.20);
    }

    .metric-label {
        color: #8994a3;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }

    .metric-value {
        color: #f4f6f8;
        font-size: 2rem;
        font-weight: 750;
        margin-top: 8px;
    }

    .metric-small {
        color: #788493;
        font-size: 0.78rem;
        margin-top: 4px;
    }

    /* Section headers */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 1.6rem;
        margin-bottom: 0.8rem;
    }

    /* Status */
    .status-box {
        border-radius: 12px;
        padding: 18px;
        margin: 10px 0;
        border: 1px solid #293341;
        background: #10161e;
    }

    .status-title {
        font-size: 1.4rem;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .status-description {
        color: #9aa5b2;
        font-size: 0.9rem;
    }

    /* Detection card */
    .detection-card {
        background: #10161e;
        border: 1px solid #252e3a;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
    }

    .detection-name {
        font-size: 1.05rem;
        font-weight: 700;
    }

    .detection-info {
        color: #8d98a7;
        font-size: 0.83rem;
        margin-top: 4px;
    }

    /* Info panel */
    .info-panel {
        background: #0f151d;
        border: 1px solid #222b36;
        border-radius: 12px;
        padding: 18px;
        margin-top: 10px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #657180;
        font-size: 0.75rem;
        padding: 35px 0 10px 0;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #303a47;
        background: #151c25;
        color: #e9edf1;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #687585;
        color: white;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Hide default menu/footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def metric_card(label, value, description=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-small">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def status_box(title, description):
    st.markdown(
        f"""
        <div class="status-box">
            <div class="status-title">{title}</div>
            <div class="status-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def clean_defect_name(name):
    return str(name).replace("_", " ").title()


def get_severity_class(severity):
    severity = str(severity).upper()

    if severity == "HIGH":
        return "HIGH"
    elif severity == "MEDIUM":
        return "MEDIUM"
    else:
        return "LOW"


def initialize_database():
    connection = sqlite3.connect(HISTORY_DB)

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            image_name TEXT,
            defect_count INTEGER,
            main_defect TEXT,
            affected_area REAL,
            average_confidence REAL,
            severity_score REAL,
            severity TEXT,
            decision TEXT
        )
        """
    )

    connection.commit()
    connection.close()


def save_history(
    image_name,
    defect_count,
    main_defect,
    affected_area,
    average_confidence,
    severity_score,
    severity,
    decision
):
    connection = sqlite3.connect(HISTORY_DB)

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO inspections (
            timestamp,
            image_name,
            defect_count,
            main_defect,
            affected_area,
            average_confidence,
            severity_score,
            severity,
            decision
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_name,
            defect_count,
            main_defect,
            affected_area,
            average_confidence,
            severity_score,
            severity,
            decision
        )
    )

    connection.commit()
    connection.close()


def load_history():
    connection = sqlite3.connect(HISTORY_DB)

    df = pd.read_sql_query(
        """
        SELECT *
        FROM inspections
        ORDER BY id DESC
        """,
        connection
    )

    connection.close()

    return df


def load_batch_data():
    if not os.path.exists(BATCH_CSV):
        return pd.DataFrame()

    try:
        df = pd.read_csv(BATCH_CSV)
    except Exception:
        return pd.DataFrame()

    # Compatibility with current CSV
    if "severity" in df.columns and "severity_level" not in df.columns:
        df["severity_level"] = df["severity"]

    return df


def calculate_image_quality(image):
    if image is None:
        return {
            "quality_status": "REJECT",
            "quality_message": "Image could not be read."
        }

    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    issues = []

    if width < 100 or height < 100:
        issues.append("Low resolution")

    if brightness < 40:
        issues.append("Too dark")

    if brightness > 220:
        issues.append("Too bright")

    if contrast < 20:
        issues.append("Low contrast")

    if blur_score < 50:
        issues.append("Possible blur")

    if issues:
        status = "REVIEW"
        message = "; ".join(issues)
    else:
        status = "PASS"
        message = "Image quality is suitable for inspection."

    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur_score": round(blur_score, 2),
        "quality_status": status,
        "quality_message": message
    }


def calculate_morphology(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)

    height, width = image.shape[:2]

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

    roi = image[y1:y2, x1:x2]

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

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
            ) / (perimeter ** 2)
        else:
            circularity = 0

    defect_width = x2 - x1
    defect_height = y2 - y1

    if defect_height > 0:
        aspect_ratio = (
            defect_width / defect_height
        )
    else:
        aspect_ratio = 0

    return {
        "area_pixels": round(defect_area, 2),
        "aspect_ratio": round(aspect_ratio, 3),
        "perimeter_pixels": round(perimeter, 2),
        "circularity": round(
            min(circularity, 1.0),
            3
        )
    }


def calculate_zone(
    bbox,
    image_width,
    image_height
):
    x1, y1, x2, y2 = bbox

    center_x = (
        x1 + x2
    ) / 2

    center_y = (
        y1 + y2
    ) / 2

    relative_x = (
        center_x /
        image_width
    )

    relative_y = (
        center_y /
        image_height
    )

    if relative_x < 1 / 3:
        column = 0
    elif relative_x < 2 / 3:
        column = 1
    else:
        column = 2

    if relative_y < 1 / 3:
        row = 0
    elif relative_y < 2 / 3:
        row = 1
    else:
        row = 2

    zone_names = [
        [
            "Top-Left",
            "Top-Center",
            "Top-Right"
        ],
        [
            "Middle-Left",
            "Center",
            "Middle-Right"
        ],
        [
            "Bottom-Left",
            "Bottom-Center",
            "Bottom-Right"
        ]
    ]

    return {
        "zone": zone_names[row][column],
        "center_x": round(center_x, 2),
        "center_y": round(center_y, 2)
    }


def calculate_severity(
    detections,
    image_area
):
    if not detections:
        return {
            "score": 0,
            "severity": "LOW",
            "decision": "PASS",
            "explanation": "No defects detected."
        }

    total_area = sum(
        d["bbox_area"]
        for d in detections
    )

    affected_percentage = (
        total_area /
        image_area
    ) * 100

    largest_percentage = max(
        d["area_percentage"]
        for d in detections
    )

    count = len(detections)

    # Area component
    if affected_percentage <= 2:
        area_score = 5
    elif affected_percentage <= 5:
        area_score = 12
    elif affected_percentage <= 10:
        area_score = 20
    elif affected_percentage <= 20:
        area_score = 26
    else:
        area_score = 30

    # Size component
    if largest_percentage <= 2:
        size_score = 4
    elif largest_percentage <= 5:
        size_score = 8
    elif largest_percentage <= 10:
        size_score = 13
    elif largest_percentage <= 20:
        size_score = 17
    else:
        size_score = 20

    # Count component
    if count == 1:
        count_score = 4
    elif count <= 3:
        count_score = 8
    elif count <= 5:
        count_score = 12
    else:
        count_score = 15

    # Location component
    location_score = 0

    for detection in detections:

        zone = detection["zone"]

        if zone == "Center":
            location_score += 10

        elif zone in [
            "Top-Center",
            "Bottom-Center",
            "Middle-Left",
            "Middle-Right"
        ]:
            location_score += 6

        else:
            location_score += 3

    location_score = min(
        location_score,
        20
    )

    # Type component
    high_risk = {
        "crazing",
        "pitted_surface",
        "rolled_in_scale"
    }

    has_high_risk = any(
        d["defect_type"]
        in high_risk
        for d in detections
    )

    type_score = (
        15
        if has_high_risk
        else 7
    )

    score = (
        area_score +
        size_score +
        count_score +
        location_score +
        type_score
    )

    if score <= 30:
        severity = "LOW"
        decision = "PASS"

    elif score <= 60:
        severity = "MEDIUM"
        decision = "INSPECT"

    else:
        severity = "HIGH"
        decision = "REJECT"

    explanation = (
        f"{count} defect(s) detected. "
        f"Estimated affected area: "
        f"{affected_percentage:.2f}%. "
        f"Largest defect: "
        f"{largest_percentage:.2f}%."
    )

    return {
        "score": round(score, 2),
        "severity": severity,
        "decision": decision,
        "affected_area": round(
            affected_percentage,
            2
        ),
        "largest_defect": round(
            largest_percentage,
            2
        ),
        "explanation": explanation
    }


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return YOLO(MODEL_PATH)


initialize_database()

model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:1.55rem;
            font-weight:750;
            margin-bottom:3px;
        ">
            InspectAI
        </div>

        <div style="
            color:#7f8a98;
            font-size:0.82rem;
            margin-bottom:25px;
        ">
            Industrial Vision Inspection
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "New Inspection",
            "Batch Inspection",
            "Inspection History",
            "Model Performance",
            "About"
        ]
    )

    st.divider()

    if model is not None:

        st.success(
            "Detection model loaded"
        )

    else:

        st.error(
            "Detection model not found"
        )

    st.caption(
        "YOLO-based steel surface defect analysis"
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">Inspection Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        AI-assisted monitoring of industrial surface defects,
        inspection decisions, and model activity.
        </div>
        """,
        unsafe_allow_html=True
    )

    batch_df = load_batch_data()
    history_df = load_history()

    # --------------------------------------------------------
    # KPI SECTION
    # --------------------------------------------------------

    total_inspections = len(batch_df)

    if total_inspections == 0:
        total_inspections = len(history_df)

    if not batch_df.empty:

        total_defects = int(
            batch_df["defect_count"].sum()
        )

        avg_confidence = (
            batch_df["average_confidence"]
            .mean()
        )

        avg_area = (
            batch_df["affected_area_percentage"]
            .mean()
        )

        if "decision" in batch_df.columns:

            reject_count = int(
                (
                    batch_df["decision"]
                    .astype(str)
                    .str.upper()
                    == "REJECT"
                ).sum()
            )

            inspect_count = int(
                (
                    batch_df["decision"]
                    .astype(str)
                    .str.upper()
                    == "INSPECT"
                ).sum()
            )

            pass_count = int(
                (
                    batch_df["decision"]
                    .astype(str)
                    .str.upper()
                    == "PASS"
                ).sum()
            )

        else:
            reject_count = 0
            inspect_count = 0
            pass_count = 0

    else:

        total_defects = 0
        avg_confidence = 0
        avg_area = 0
        reject_count = 0
        inspect_count = 0
        pass_count = 0

    cols = st.columns(4)

    with cols[0]:
        metric_card(
            "Total Inspections",
            total_inspections,
            "Images analyzed"
        )

    with cols[1]:
        metric_card(
            "Defects Detected",
            total_defects,
            "Across inspected images"
        )

    with cols[2]:
        metric_card(
            "Average Confidence",
            f"{avg_confidence:.2f}",
            "Model prediction confidence"
        )

    with cols[3]:
        metric_card(
            "Affected Area",
            f"{avg_area:.2f}%",
            "Average estimated area"
        )

    # --------------------------------------------------------
    # DECISION SUMMARY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Inspection Decisions</div>',
        unsafe_allow_html=True
    )

    decision_cols = st.columns(3)

    with decision_cols[0]:
        metric_card(
            "PASS",
            pass_count,
            "No critical inspection issue"
        )

    with decision_cols[1]:
        metric_card(
            "INSPECT",
            inspect_count,
            "Manual review recommended"
        )

    with decision_cols[2]:
        metric_card(
            "REJECT",
            reject_count,
            "High severity classification"
        )

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    if not batch_df.empty:

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            st.markdown(
                '<div class="section-title">Defects by Type</div>',
                unsafe_allow_html=True
            )

            defect_counts = {}

            for value in batch_df["main_defect"].dropna():

                value = str(value)

                if value.lower() == "none":
                    continue

                defect_counts[value] = (
                    defect_counts.get(value, 0)
                    + 1
                )

            if defect_counts:

                chart_df = pd.DataFrame(
                    {
                        "Defect": list(
                            defect_counts.keys()
                        ),
                        "Images": list(
                            defect_counts.values()
                        )
                    }
                )

                st.bar_chart(
                    chart_df.set_index("Defect")
                )

            else:

                st.info(
                    "No defect categories available."
                )

        with chart_col2:

            st.markdown(
                '<div class="section-title">Severity Distribution</div>',
                unsafe_allow_html=True
            )

            if "severity" in batch_df.columns:

                severity_counts = (
                    batch_df["severity"]
                    .value_counts()
                )

                st.bar_chart(
                    severity_counts
                )

        # ----------------------------------------------------
        # RECENT BATCH RESULTS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Recent Batch Results</div>',
            unsafe_allow_html=True
        )

        display_columns = [
            "image",
            "defect_count",
            "main_defect",
            "average_confidence",
            "affected_area_percentage",
            "severity",
            "decision"
        ]

        available_columns = [
            c
            for c in display_columns
            if c in batch_df.columns
        ]

        recent = batch_df[
            available_columns
        ].head(10)

        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No batch inspection data available yet. "
            "Run Batch Inspection to populate the dashboard."
        )


# ============================================================
# NEW INSPECTION
# ============================================================

elif page == "New Inspection":

    st.markdown(
        '<div class="main-title">New Inspection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Upload a steel-surface image for AI-powered defect
        detection and severity analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

    if model is None:

        st.error(
            "Model not found. Please verify the best.pt path."
        )

    else:

        uploaded_file = st.file_uploader(
            "Upload inspection image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "bmp"
            ]
        )

        if uploaded_file is not None:

            pil_image = Image.open(
                uploaded_file
            ).convert("RGB")

            image = np.array(
                pil_image
            )

            image_bgr = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )

            left, right = st.columns(
                [1.1, 0.9]
            )

            with left:

                st.image(
                    pil_image,
                    caption="Input Inspection Image",
                    use_container_width=True
                )

            with right:

                quality = calculate_image_quality(
                    image_bgr
                )

                st.markdown(
                    '<div class="section-title">Image Quality</div>',
                    unsafe_allow_html=True
                )

                q1, q2 = st.columns(2)

                with q1:
                    metric_card(
                        "Resolution",
                        f"{quality['width']} × {quality['height']}",
                        "Image dimensions"
                    )

                with q2:
                    metric_card(
                        "Brightness",
                        quality["brightness"],
                        "Mean grayscale intensity"
                    )

                q3, q4 = st.columns(2)

                with q3:
                    metric_card(
                        "Contrast",
                        quality["contrast"],
                        "Intensity variation"
                    )

                with q4:
                    metric_card(
                        "Blur Score",
                        quality["blur_score"],
                        "Laplacian variance"
                    )

                if quality["quality_status"] == "PASS":

                    status_box(
                        "QUALITY: PASS",
                        quality["quality_message"]
                    )

                else:

                    status_box(
                        "QUALITY: REVIEW",
                        quality["quality_message"]
                    )

            st.divider()

            with st.spinner(
                "Running AI inspection..."
            ):

                results = model.predict(
                    source=image_bgr,
                    conf=CONFIDENCE_THRESHOLD,
                    imgsz=IMAGE_SIZE,
                    verbose=False
                )

            result = results[0]

            detections = []

            image_height, image_width = (
                image_bgr.shape[:2]
            )

            image_area = (
                image_width *
                image_height
            )

            if result.boxes is not None:

                boxes = result.boxes

                for i in range(
                    len(boxes)
                ):

                    box = (
                        boxes.xyxy[i]
                        .cpu()
                        .numpy()
                    )

                    x1, y1, x2, y2 = box

                    confidence = float(
                        boxes.conf[i]
                        .cpu()
                        .numpy()
                    )

                    class_id = int(
                        boxes.cls[i]
                        .cpu()
                        .numpy()
                    )

                    class_name = model.names[
                        class_id
                    ]

                    width = max(
                        0,
                        x2 - x1
                    )

                    height = max(
                        0,
                        y2 - y1
                    )

                    bbox_area = (
                        width *
                        height
                    )

                    area_percentage = (
                        bbox_area /
                        image_area
                    ) * 100

                    morphology = (
                        calculate_morphology(
                            image_bgr,
                            box
                        )
                    )

                    zone = calculate_zone(
                        box,
                        image_width,
                        image_height
                    )

                    detections.append(
                        {
                            "defect_type": class_name,
                            "confidence": confidence,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "width": width,
                            "height": height,
                            "bbox_area": bbox_area,
                            "area_percentage": area_percentage,
                            "aspect_ratio": morphology[
                                "aspect_ratio"
                            ],
                            "perimeter": morphology[
                                "perimeter_pixels"
                            ],
                            "circularity": morphology[
                                "circularity"
                            ],
                            "zone": zone[
                                "zone"
                            ]
                        }
                    )

            severity = calculate_severity(
                detections,
                image_area
            )

            # Low confidence forces manual review
            low_confidence = [
                d
                for d in detections
                if d["confidence"] < 0.30
            ]

            moderate_confidence = [
                d
                for d in detections
                if 0.30 <= d["confidence"] < 0.60
            ]

            final_decision = (
                severity["decision"]
            )

            if low_confidence:
                final_decision = "INSPECT"

            if (
                quality["quality_status"]
                == "REVIEW"
                and final_decision
                == "PASS"
            ):
                final_decision = "INSPECT"

            # ------------------------------------------------
            # RESULT HEADER
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">Inspection Result</div>',
                unsafe_allow_html=True
            )

            r1, r2, r3, r4 = st.columns(4)

            with r1:
                metric_card(
                    "Defects",
                    len(detections),
                    "Detected objects"
                )

            with r2:
                metric_card(
                    "Severity Score",
                    severity["score"],
                    "Project-defined score"
                )

            with r3:
                metric_card(
                    "Severity",
                    severity["severity"],
                    "Overall severity level"
                )

            with r4:
                metric_card(
                    "Decision",
                    final_decision,
                    "Inspection disposition"
                )

            # ------------------------------------------------
            # DECISION MESSAGE
            # ------------------------------------------------

            if final_decision == "PASS":

                status_box(
                    "FINAL DECISION: PASS",
                    "No high-severity defect condition was identified."
                )

            elif final_decision == "INSPECT":

                status_box(
                    "FINAL DECISION: INSPECT",
                    "Manual review is recommended before accepting the material."
                )

            else:

                status_box(
                    "FINAL DECISION: REJECT",
                    "The project severity rules classified this inspection as high severity."
                )

            # ------------------------------------------------
            # CONFIDENCE WARNINGS
            # ------------------------------------------------

            if low_confidence:

                st.warning(
                    f"{len(low_confidence)} low-confidence "
                    "prediction(s) detected. Manual verification is recommended."
                )

            elif moderate_confidence:

                st.info(
                    f"{len(moderate_confidence)} prediction(s) "
                    "have moderate confidence."
                )

            # ------------------------------------------------
            # ANNOTATED IMAGE
            # ------------------------------------------------

            annotated = result.plot()

            annotated_rgb = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

            st.markdown(
                '<div class="section-title">Defect Localization</div>',
                unsafe_allow_html=True
            )

            st.image(
                annotated_rgb,
                caption="YOLO Detection Output",
                use_container_width=True
            )

            # ------------------------------------------------
            # DETECTION DETAILS
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">Detection Details</div>',
                unsafe_allow_html=True
            )

            if detections:

                for index, detection in enumerate(
                    detections,
                    start=1
                ):

                    confidence = (
                        detection["confidence"]
                    )

                    confidence_status = (
                        "LOW"
                        if confidence < 0.30
                        else (
                            "MODERATE"
                            if confidence < 0.60
                            else "HIGHER"
                        )
                    )

                    st.markdown(
                        f"""
                        <div class="detection-card">

                        <div class="detection-name">
                        Detection {index} —
                        {clean_defect_name(
                            detection["defect_type"]
                        )}
                        </div>

                        <div class="detection-info">
                        Confidence: {confidence:.2f}
                        &nbsp;&nbsp;|&nbsp;&nbsp;
                        Confidence Status: {confidence_status}
                        &nbsp;&nbsp;|&nbsp;&nbsp;
                        Estimated Area: {detection["area_percentage"]:.2f}%
                        </div>

                        <div class="detection-info">
                        Zone: {detection["zone"]}
                        &nbsp;&nbsp;|&nbsp;&nbsp;
                        Aspect Ratio: {detection["aspect_ratio"]:.3f}
                        &nbsp;&nbsp;|&nbsp;&nbsp;
                        Circularity: {detection["circularity"]:.3f}
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                details_df = pd.DataFrame(
                    [
                        {
                            "Defect":
                                clean_defect_name(
                                    d["defect_type"]
                                ),
                            "Confidence":
                                round(
                                    d["confidence"],
                                    3
                                ),
                            "Area %":
                                round(
                                    d["area_percentage"],
                                    2
                                ),
                            "Zone":
                                d["zone"],
                            "Aspect Ratio":
                                round(
                                    d["aspect_ratio"],
                                    3
                                ),
                            "Circularity":
                                round(
                                    d["circularity"],
                                    3
                                )
                        }
                        for d in detections
                    ]
                )

                st.dataframe(
                    details_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.success(
                    "No defects were detected at the configured confidence threshold."
                )

            # ------------------------------------------------
            # SAVE HISTORY
            # ------------------------------------------------

            if detections:

                main_defect = max(
                    detections,
                    key=lambda x: x["confidence"]
                )["defect_type"]

                average_confidence = np.mean(
                    [
                        d["confidence"]
                        for d in detections
                    ]
                )

                affected_area = (
                    severity.get(
                        "affected_area",
                        0
                    )
                )

            else:

                main_defect = "None"
                average_confidence = 0
                affected_area = 0

            save_history(
                image_name=uploaded_file.name,
                defect_count=len(detections),
                main_defect=main_defect,
                affected_area=affected_area,
                average_confidence=float(
                    average_confidence
                ),
                severity_score=severity["score"],
                severity=severity["severity"],
                decision=final_decision
            )

            # ------------------------------------------------
            # RECOMMENDATION
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">Inspection Explanation</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="info-panel">

                <b>Analysis:</b><br>
                {severity["explanation"]}

                <br><br>

                <b>Recommended workflow:</b><br>
                {"Proceed with normal processing."
                if final_decision == "PASS"
                else "Perform manual inspection and verify the detected region."
                }

                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # DOWNLOAD RESULTS
            # ------------------------------------------------

            if detections:

                export_df = pd.DataFrame(
                    [
                        {
                            "image":
                                uploaded_file.name,
                            "defect_type":
                                d["defect_type"],
                            "confidence":
                                round(
                                    d["confidence"],
                                    4
                                ),
                            "area_percentage":
                                round(
                                    d["area_percentage"],
                                    4
                                ),
                            "aspect_ratio":
                                d["aspect_ratio"],
                            "circularity":
                                d["circularity"],
                            "zone":
                                d["zone"],
                            "severity_score":
                                severity["score"],
                            "severity":
                                severity["severity"],
                            "decision":
                                final_decision
                        }
                        for d in detections
                    ]
                )

                csv_data = export_df.to_csv(
                    index=False
                )

                st.download_button(
                    "Download Inspection CSV",
                    data=csv_data,
                    file_name="inspectai_inspection.csv",
                    mime="text/csv"
                )


# ============================================================
# BATCH INSPECTION
# ============================================================

elif page == "Batch Inspection":

    st.markdown(
        '<div class="main-title">Batch Inspection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Review inspection results across the validation dataset
        and analyze aggregate defect patterns.
        </div>
        """,
        unsafe_allow_html=True
    )

    batch_df = load_batch_data()

    if batch_df.empty:

        st.warning(
            "No batch results found."
        )

        st.code(
            "python src/batch_inspector.py",
            language="text"
        )

    else:

        # ----------------------------------------------------
        # BATCH KPIs
        # ----------------------------------------------------

        total = len(batch_df)

        defects = int(
            batch_df["defect_count"].sum()
        )

        avg_conf = (
            batch_df[
                "average_confidence"
            ].mean()
        )

        avg_area = (
            batch_df[
                "affected_area_percentage"
            ].mean()
        )

        cols = st.columns(4)

        with cols[0]:
            metric_card(
                "Images",
                total,
                "Batch size"
            )

        with cols[1]:
            metric_card(
                "Defects",
                defects,
                "Total detections"
            )

        with cols[2]:
            metric_card(
                "Avg Confidence",
                f"{avg_conf:.2f}",
                "Prediction confidence"
            )

        with cols[3]:
            metric_card(
                "Avg Affected Area",
                f"{avg_area:.2f}%",
                "Estimated area"
            )

        st.divider()

        # ----------------------------------------------------
        # BATCH CHARTS
        # ----------------------------------------------------

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                '<div class="section-title">Decision Distribution</div>',
                unsafe_allow_html=True
            )

            if "decision" in batch_df.columns:

                decision_counts = (
                    batch_df["decision"]
                    .value_counts()
                )

                st.bar_chart(
                    decision_counts
                )

        with c2:

            st.markdown(
                '<div class="section-title">Defect Distribution</div>',
                unsafe_allow_html=True
            )

            defect_counts = (
                batch_df["main_defect"]
                .value_counts()
            )

            st.bar_chart(
                defect_counts
            )

        # ----------------------------------------------------
        # FILTER
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Filter Results</div>',
            unsafe_allow_html=True
        )

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:

            if "decision" in batch_df.columns:

                decisions = [
                    "ALL"
                ] + sorted(
                    batch_df[
                        "decision"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                selected_decision = st.selectbox(
                    "Decision",
                    decisions
                )

            else:

                selected_decision = "ALL"

        with filter_col2:

            if "severity" in batch_df.columns:

                severities = [
                    "ALL"
                ] + sorted(
                    batch_df[
                        "severity"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                selected_severity = st.selectbox(
                    "Severity",
                    severities
                )

            else:

                selected_severity = "ALL"

        filtered_df = batch_df.copy()

        if (
            selected_decision
            != "ALL"
        ):

            filtered_df = filtered_df[
                filtered_df["decision"]
                == selected_decision
            ]

        if (
            selected_severity
            != "ALL"
        ):

            filtered_df = filtered_df[
                filtered_df["severity"]
                == selected_severity
            ]

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = filtered_df.to_csv(
            index=False
        )

        st.download_button(
            "Download Batch Report",
            data=csv_data,
            file_name="inspectai_batch_report.csv",
            mime="text/csv"
        )


# ============================================================
# INSPECTION HISTORY
# ============================================================

elif page == "Inspection History":

    st.markdown(
        '<div class="main-title">Inspection History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Persistent inspection records stored locally using SQLite.
        </div>
        """,
        unsafe_allow_html=True
    )

    history_df = load_history()

    if history_df.empty:

        st.info(
            "No inspection history available yet."
        )

    else:

        cols = st.columns(4)

        with cols[0]:
            metric_card(
                "Records",
                len(history_df),
                "Stored inspections"
            )

        with cols[1]:

            avg_score = (
                history_df[
                    "severity_score"
                ].mean()
            )

            metric_card(
                "Avg Severity",
                f"{avg_score:.1f}",
                "Project-defined score"
            )

        with cols[2]:

            total_defects = int(
                history_df[
                    "defect_count"
                ].sum()
            )

            metric_card(
                "Total Defects",
                total_defects,
                "Historical detections"
            )

        with cols[3]:

            avg_conf = (
                history_df[
                    "average_confidence"
                ].mean()
            )

            metric_card(
                "Avg Confidence",
                f"{avg_conf:.2f}",
                "Historical average"
            )

        st.divider()

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(
            index=False
        )

        st.download_button(
            "Download Inspection History",
            data=csv_data,
            file_name="inspectai_history.csv",
            mime="text/csv"
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="main-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Evaluation metrics for the trained YOLO defect detector.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    cols = st.columns(4)

    with cols[0]:
        metric_card(
            "Precision",
            f"{MODEL_METRICS['Precision']:.1f}%",
            "Detection precision"
        )

    with cols[1]:
        metric_card(
            "Recall",
            f"{MODEL_METRICS['Recall']:.1f}%",
            "Detection recall"
        )

    with cols[2]:
        metric_card(
            "mAP@50",
            f"{MODEL_METRICS['mAP@50']:.1f}%",
            "IoU threshold 0.50"
        )

    with cols[3]:
        metric_card(
            "mAP@50-95",
            f"{MODEL_METRICS['mAP@50-95']:.1f}%",
            "COCO-style range"
        )

    # --------------------------------------------------------
    # CLASS METRICS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Class-wise Performance</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        CLASS_METRICS,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # CLASS CHART
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">mAP@50 by Defect Class</div>',
        unsafe_allow_html=True
    )

    map_chart = (
        CLASS_METRICS[
            [
                "Defect",
                "mAP@50"
            ]
        ]
        .set_index("Defect")
    )

    st.bar_chart(
        map_chart
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Evaluation Notes</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-panel">

        <b>Dataset:</b> NEU-DET steel surface defect dataset<br><br>

        <b>Defect categories:</b>
        crazing, inclusion, patches, pitted surface,
        rolled-in scale, scratches<br><br>

        <b>Training configuration:</b>
        YOLO11n, 15 epochs, image size 320,
        batch size 8, CPU training<br><br>

        <b>Interpretation:</b>
        The model demonstrates different detection performance
        across defect categories. Thin and distributed defects
        such as crazing remain challenging for the current model.
        This dashboard reports the measured evaluation metrics
        rather than treating them as a single overall accuracy value.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Important Limitation</div>',
        unsafe_allow_html=True
    )

    st.warning(
        "The severity score and PASS/INSPECT/REJECT rules "
        "are project-defined heuristics. They are not an "
        "industrial certification or universal engineering standard."
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.markdown(
        '<div class="main-title">About InspectAI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        AI-powered industrial surface inspection and
        decision-support prototype.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-panel">

        <h3>System Overview</h3>

        InspectAI analyzes industrial surface images using
        computer vision and object detection techniques.

        <br><br>

        <b>Pipeline</b>

        <br><br>

        Image Input
        → Image Quality Gate
        → YOLO Defect Detection
        → Defect Localization
        → Morphological Analysis
        → Zone Analysis
        → Severity Engine
        → Confidence Review
        → Final Decision

        <br><br>

        <b>Supported defects</b>

        <br>

        • Crazing<br>
        • Inclusion<br>
        • Patches<br>
        • Pitted Surface<br>
        • Rolled-in Scale<br>
        • Scratches

        <br><br>

        <b>Technology Stack</b>

        <br>

        Python • OpenCV • YOLO • NumPy • Pandas • Streamlit • SQLite

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">System Capabilities</div>',
        unsafe_allow_html=True
    )

    capabilities = pd.DataFrame(
        [
            [
                "Object Detection",
                "YOLO-based localization of surface defects"
            ],
            [
                "Quality Gate",
                "Brightness, contrast, resolution and blur checks"
            ],
            [
                "Morphology",
                "Area, aspect ratio, perimeter and circularity"
            ],
            [
                "Zone Analysis",
                "Spatial location of detected defects"
            ],
            [
                "Severity Engine",
                "Project-defined multi-factor severity scoring"
            ],
            [
                "Manual Review",
                "Low-confidence predictions are flagged"
            ],
            [
                "Batch Analysis",
                "Aggregate analysis of multiple images"
            ],
            [
                "History",
                "SQLite-backed inspection records"
            ]
        ],
        columns=[
            "Component",
            "Function"
        ]
    )

    st.dataframe(
        capabilities,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">Project Disclaimer</div>',
        unsafe_allow_html=True
    )

    st.info(
        "InspectAI is an academic/prototype computer vision system. "
        "Its model predictions and severity rules should be validated "
        "against domain-specific industrial standards and expert "
        "inspection procedures before any real production deployment."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    InspectAI — AI-Powered Industrial Defect Detection & Severity Analysis
    </div>
    """,
    unsafe_allow_html=True
)
