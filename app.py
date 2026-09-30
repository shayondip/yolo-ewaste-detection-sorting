from pathlib import Path
from collections import Counter

import streamlit as st
from PIL import Image
from ultralytics import YOLO

from sorting_rules import get_sorting_recommendation


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI E-Waste Detection & Sorting",
    page_icon="♻️",
    layout="wide",
)

MODEL_PATH = Path("models/best.pt")


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.resolve()}"
        )

    return YOLO(str(MODEL_PATH))


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("♻️ AI E-Waste Detection & Sorting")
st.markdown(
    """
    **YOLO11n-based computer vision system** for detecting waste objects
    and providing a rule-based sorting recommendation.
    """
)

st.divider()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model Information")

    st.write("**Model:** YOLO11n")
    st.write("**Classes:** 12")
    st.write("**Input Size:** 640 × 640")

    st.divider()

    st.subheader("Test Performance")
    st.metric("Precision", "82.6%")
    st.metric("Recall", "89.9%")
    st.metric("mAP@50", "88.8%")
    st.metric("mAP@50–95", "73.1%")

    st.caption(
        "These metrics are from the current test evaluation and may change "
        "after dataset cleanup/retraining."
    )


# ---------------------------------------------------------
# Model initialization
# ---------------------------------------------------------
try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()


# ---------------------------------------------------------
# Upload section
# ---------------------------------------------------------
st.subheader("📤 Upload an E-Waste Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Upload an image containing one or more waste objects.",
)


# ---------------------------------------------------------
# Confidence threshold
# ---------------------------------------------------------
confidence_threshold = st.slider(
    "Detection confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05,
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Input Image")
        st.image(image, use_container_width=True)

    with st.spinner("Running YOLO detection..."):

        results = model.predict(
            source=image,
            imgsz=640,
            conf=confidence_threshold,
            device="cpu",
            verbose=False,
        )

    result = results[0]

    # Annotated image
    annotated = result.plot()

    with col2:
        st.subheader("🔍 Detection Result")
        st.image(annotated, channels="BGR", use_container_width=True)

    st.divider()

    # -----------------------------------------------------
    # No detections
    # -----------------------------------------------------
    if result.boxes is None or len(result.boxes) == 0:
        st.warning(
            "No object was detected above the selected confidence threshold."
        )
        st.stop()

    # -----------------------------------------------------
    # Extract detections
    # -----------------------------------------------------
    detections = []

    for box in result.boxes:

        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        class_name = model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        sorting_info = get_sorting_recommendation(class_name)

        detections.append(
            {
                "class": class_name,
                "confidence": confidence,
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
                "category": sorting_info["category"],
                "bin": sorting_info["bin"],
                "instruction": sorting_info["instruction"],
            }
        )

    # -----------------------------------------------------
    # Summary metrics
    # -----------------------------------------------------
    st.subheader("📊 Detection Summary")

    class_counts = Counter(item["class"] for item in detections)

    total_objects = len(detections)
    unique_classes = len(class_counts)
    mean_confidence = sum(
        item["confidence"] for item in detections
    ) / total_objects

    m1, m2, m3 = st.columns(3)

    m1.metric("Total Objects", total_objects)
    m2.metric("Unique Classes", unique_classes)
    m3.metric("Mean Confidence", f"{mean_confidence:.1%}")

    # -----------------------------------------------------
    # Detected objects
    # -----------------------------------------------------
    st.subheader("🎯 Detected Objects")

    for index, item in enumerate(detections, start=1):

        st.markdown(f"### {index}. {item['class']}")

        c1, c2, c3 = st.columns(3)

        c1.write(f"**Confidence:** {item['confidence']:.1%}")
        c2.write(f"**Category:** {item['category']}")
        c3.write(f"**Recommended Bin:** {item['bin']}")

        st.info(f"♻️ {item['instruction']}")

        st.caption(
            "Bounding box: "
            f"({item['x1']}, {item['y1']}) → "
            f"({item['x2']}, {item['y2']})"
        )

        st.divider()

    # -----------------------------------------------------
    # Class count table
    # -----------------------------------------------------
    st.subheader("📋 Class Count")

    sorted_counts = dict(
        sorted(class_counts.items(), key=lambda x: x[0])
    )

    for class_name, count in sorted_counts.items():
        st.write(f"**{class_name}:** {count}")