import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "results/detections/detection_results.csv"

OUTPUT_DIR = "results/severity"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "inspection_result.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PROJECT-DEFINED SEVERITY WEIGHTS
# ============================================================

AREA_WEIGHT = 30
SIZE_WEIGHT = 20
COUNT_WEIGHT = 15
LOCATION_WEIGHT = 20
TYPE_WEIGHT = 15


# Defect types treated as higher-risk within this project
HIGH_RISK_DEFECTS = {
    "crazing",
    "pitted_surface",
    "rolled-in_scale"
}


# ============================================================
# LOAD DETECTION DATA
# ============================================================

if not os.path.exists(INPUT_FILE):

    print("ERROR: Detection CSV not found.")
    print(f"Expected: {INPUT_FILE}")
    print()
    print("Run defect_analyzer.py first.")
    exit()


df = pd.read_csv(INPUT_FILE)


print("=" * 65)
print("INSPECTAI EXPLAINABLE SEVERITY ENGINE")
print("=" * 65)

print()
print(f"Input file: {INPUT_FILE}")
print(f"Detection records: {len(df)}")


# ============================================================
# HANDLE NO DETECTIONS
# ============================================================

if len(df) == 0:

    result = pd.DataFrame([{
        "severity_score": 0,
        "severity": "LOW",
        "decision": "PASS",
        "defect_count": 0,
        "affected_area_percentage": 0,
        "average_confidence": 0,
        "main_defect": "None",
        "explanation": "No defects detected."
    }])

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("No defects detected.")
    print("Decision: PASS")
    print(f"Result saved to: {OUTPUT_FILE}")

    exit()


# ============================================================
# BASIC STATISTICS
# ============================================================

defect_count = len(df)

average_confidence = df[
    "confidence"
].mean()


# ============================================================
# AFFECTED AREA
# ============================================================

# Bounding-box area percentage
#
# Note:
# Overlapping bounding boxes can cause the total to be
# higher than the true affected surface area.

affected_area = df[
    "area_percentage"
].sum()

affected_area = min(
    affected_area,
    100
)


# ============================================================
# LARGEST DEFECT
# ============================================================

largest_defect = df[
    "area_percentage"
].max()


# ============================================================
# DEFECT TYPE
# ============================================================

defect_counts = df[
    "defect_type"
].value_counts()

main_defect = defect_counts.index[0]


# ============================================================
# SCORE 1 — AREA IMPACT
# ============================================================

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


# ============================================================
# SCORE 2 — DEFECT SIZE
# ============================================================

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


# ============================================================
# SCORE 3 — DEFECT COUNT
# ============================================================

if defect_count == 1:

    count_score = 4

elif defect_count <= 3:

    count_score = 8

elif defect_count <= 5:

    count_score = 12

else:

    count_score = 15


# ============================================================
# SCORE 4 — LOCATION
# ============================================================

location_score = 0

location_reasons = []


if "zone" in df.columns:

    for zone in df["zone"]:

        if zone == "Center":

            location_score += 10

            location_reasons.append(
                "Defect detected in center region"
            )

        elif zone in [
            "Top-Center",
            "Middle-Left",
            "Middle-Right",
            "Bottom-Center"
        ]:

            location_score += 6

        else:

            location_score += 3


    # Prevent exceeding location weight
    location_score = min(
        location_score,
        LOCATION_WEIGHT
    )


else:

    location_score = 0


# ============================================================
# SCORE 5 — DEFECT TYPE
# ============================================================

high_risk_detected = []

for defect in df["defect_type"]:

    if defect in HIGH_RISK_DEFECTS:

        high_risk_detected.append(
            defect
        )


if len(high_risk_detected) > 0:

    type_score = TYPE_WEIGHT

else:

    type_score = 7


# ============================================================
# CONFIDENCE / UNCERTAINTY
# ============================================================

low_confidence_count = len(
    df[df["confidence"] < 0.40]
)

moderate_confidence_count = len(
    df[
        (df["confidence"] >= 0.40)
        & (df["confidence"] < 0.60)
    ]
)


# Confidence does NOT directly represent physical severity.
# It is used to identify uncertain detections that may require
# manual review.


# ============================================================
# TOTAL SEVERITY SCORE
# ============================================================

severity_score = (
    area_score
    + size_score
    + count_score
    + location_score
    + type_score
)


severity_score = min(
    round(severity_score, 2),
    100
)


# ============================================================
# SEVERITY LEVEL
# ============================================================

if severity_score <= 30:

    severity = "LOW"

elif severity_score <= 60:

    severity = "MEDIUM"

else:

    severity = "HIGH"


# ============================================================
# DECISION
# ============================================================

if severity == "LOW":

    decision = "PASS"

elif severity == "MEDIUM":

    decision = "INSPECT"

else:

    decision = "REJECT"


# ============================================================
# EXPLAINABLE REASONS
# ============================================================

reasons = []


# Area explanation
if affected_area > 10:

    reasons.append(
        "Large affected surface area"
    )

elif affected_area > 5:

    reasons.append(
        "Moderate affected surface area"
    )


# Size explanation
if largest_defect > 10:

    reasons.append(
        "Large individual defect detected"
    )


# Count explanation
if defect_count >= 5:

    reasons.append(
        "Multiple defects detected"
    )

elif defect_count >= 3:

    reasons.append(
        "Several defects detected"
    )


# Type explanation
if high_risk_detected:

    unique_high_risk = sorted(
        set(high_risk_detected)
    )

    reasons.append(
        "Higher-risk defect type detected: "
        + ", ".join(unique_high_risk)
    )


# Location explanation
if location_reasons:

    reasons.extend(
        sorted(set(location_reasons))
    )


# Confidence explanation
if low_confidence_count > 0:

    reasons.append(
        f"{low_confidence_count} "
        "low-confidence detection(s) require review"
    )

elif moderate_confidence_count > 0:

    reasons.append(
        f"{moderate_confidence_count} "
        "moderate-confidence detection(s)"
    )


if not reasons:

    reasons.append(
        "Defect characteristics remain within "
        "lower severity thresholds"
    )


# ============================================================
# RECOMMENDATION
# ============================================================

if decision == "PASS":

    recommendation = (
        "No immediate intervention indicated "
        "by the project-defined severity rules."
    )

elif decision == "INSPECT":

    recommendation = (
        "Perform manual inspection before "
        "final acceptance."
    )

else:

    recommendation = (
        "Hold the component for detailed inspection "
        "or rejection according to the inspection workflow."
    )


# ============================================================
# CREATE RESULT
# ============================================================

result = pd.DataFrame([{

    "severity_score": severity_score,

    "severity": severity,

    "decision": decision,

    "defect_count": defect_count,

    "affected_area_percentage": round(
        affected_area,
        2
    ),

    "largest_defect_percentage": round(
        largest_defect,
        2
    ),

    "average_confidence": round(
        average_confidence,
        3
    ),

    "main_defect": main_defect,

    # Individual score components
    "area_score": area_score,

    "size_score": size_score,

    "count_score": count_score,

    "location_score": location_score,

    "type_score": type_score,

    # Uncertainty information
    "low_confidence_detections": low_confidence_count,

    "moderate_confidence_detections":
        moderate_confidence_count,

    # Explainability
    "explanation": " | ".join(
        reasons
    ),

    "recommendation": recommendation
}])


# ============================================================
# SAVE RESULT
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY FINAL RESULT
# ============================================================

print()
print("=" * 65)
print("INSPECTION RESULT")
print("=" * 65)

print(
    f"Severity Score : "
    f"{severity_score}/100"
)

print(
    f"Severity       : "
    f"{severity}"
)

print(
    f"Decision        : "
    f"{decision}"
)

print(
    f"Defect Count    : "
    f"{defect_count}"
)

print(
    f"Affected Area   : "
    f"{affected_area:.2f}%"
)

print(
    f"Largest Defect  : "
    f"{largest_defect:.2f}%"
)

print(
    f"Avg Confidence  : "
    f"{average_confidence:.3f}"
)

print(
    f"Main Defect     : "
    f"{main_defect}"
)


# ============================================================
# SCORE BREAKDOWN
# ============================================================

print()
print("SCORE BREAKDOWN")
print("-" * 65)

print(
    f"Area Impact : "
    f"{area_score}/{AREA_WEIGHT}"
)

print(
    f"Defect Size : "
    f"{size_score}/{SIZE_WEIGHT}"
)

print(
    f"Defect Count: "
    f"{count_score}/{COUNT_WEIGHT}"
)

print(
    f"Location    : "
    f"{location_score}/{LOCATION_WEIGHT}"
)

print(
    f"Defect Type : "
    f"{type_score}/{TYPE_WEIGHT}"
)


# ============================================================
# EXPLANATION
# ============================================================

print()
print("WHY THIS DECISION?")
print("-" * 65)

for reason in reasons:

    print(f"- {reason}")


print()
print("RECOMMENDATION")
print("-" * 65)

print(recommendation)


print()
print(
    f"Result saved to: "
    f"{OUTPUT_FILE}"
)

print()
print("=" * 65)
print("SEVERITY ANALYSIS COMPLETE")
print("=" * 65)