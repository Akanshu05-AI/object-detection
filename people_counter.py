import streamlit as st
import cv2
import cvzone
import math
import time
from ultralytics import YOLO
from sort import Sort
import numpy as np
import tempfile
import os

# Load the YOLO model and class names (load only once)
@st.cache_resource
def load_model():
    model = YOLO("../Yolo_Weights/yolov8n.pt")
    return model

@st.cache_data
def load_class_names():
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]
    return classNames

model = load_model()
classNames = load_class_names()

def process_video(video_path, mask_path, progress_bar):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return None, None

    mask_img = cv2.imread(mask_path)
    if mask_img is None:
        st.error("Error: Could not read mask image.")
        cap.release()
        return None, None
    mask_img = cv2.resize(mask_img, (1280, 720))

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    limitsUp = [103, 161, 296, 161]
    limitsDown = [527, 489, 735, 489]

    totalCountUp = []
    totalCountDown = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        success, img = cap.read()
        if not success:
            break

        imgregion = cv2.bitwise_and(img, mask_img)
        graphics_path = r"D:\minor_project\graphics-1.png"  # Use raw string for Windows path
        if os.path.exists(graphics_path):
            imgGraphics = cv2.imread(graphics_path, cv2.IMREAD_UNCHANGED)
            if imgGraphics is not None:
                img = cvzone.overlayPNG(img, imgGraphics, (730, 260))

        results = model(imgregion, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0].item() * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass == "person" and conf > 0.3:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)

        cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
        cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

        for result in resultsTracker:
            x1, y1, x2, y2, id = map(int, result)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
            cvzone.putTextRect(img, f"{int(id)}", (max(0, x1), max(35, y1)), scale=1, thickness=2, offset=3)
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
                if id not in totalCountUp:
                    totalCountUp.append(id)
                    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

            if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
                if id not in totalCountDown:
                    totalCountDown.append(id)
                    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

        cv2.putText(img, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
        cv2.putText(img, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

        progress_bar.progress((cap.get(cv2.CAP_PROP_POS_FRAMES) / frame_count) if frame_count > 0 else 1.0)

    cap.release()
    return len(totalCountUp), len(totalCountDown)

def main():
    st.title("People Counter App")

    video_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
    mask_file = st.file_uploader("Upload Mask Image", type=["png", "jpg", "jpeg"])

    if video_file is not None and mask_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video_file:
            tmp_video_file.write(video_file.read())
            video_path = tmp_video_file.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(mask_file.name)[1]) as tmp_mask_file:
            tmp_mask_file.write(mask_file.read())
            mask_path = tmp_mask_file.name

        st.subheader("Uploaded Video and Processing Status")
        col1, col2 = st.columns(2)

        with col1:
            st.info("Uploaded Video")
            st.video(video_file)

        with col2:
            st.info("Processing Video...")
            progress_bar = st.progress(0.0)
            up_count, down_count = process_video(video_path, mask_path, progress_bar)
            progress_bar.empty()

            if up_count is not None and down_count is not None:
                st.success("Video processing complete!")
                st.metric("People Going Up", up_count)
                st.metric("People Going Down", down_count)
            else:
                st.warning("No people were counted or processing failed.")

        os.unlink(video_path)
        os.unlink(mask_path)

if __name__ == "__main__":
    main()