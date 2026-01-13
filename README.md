# Object Detection & Counting Suite
This is a collection of Python scripts I built to handle real-time object detection and tracking. The project focuses on two main use cases: monitoring vehicle traffic and tracking pedestrian movement using computer vision.

### What’s inside:
* **`car_counter.py`**: Uses a detection mask and a coordinate-based line to count cars moving in a specific direction.
* **`people_counter.py`**: Similar logic but tuned for human detection and tracking in tighter spaces.
* **`object_detection.py`**: The base script for general detection tasks.

### 🛠 How I built it
The project is built on **Python** and uses **YOLOv8** for the heavy lifting (detection). I used **OpenCV** for the video processing and **CVZone** to help with the UI overlays like the counter boxes and lines.

**Dependencies you'll need:**
* `ultralytics` (for YOLOv8)
* `opencv-python`
* `cvzone`
* `numpy`

### 🚀 How to run it
1. Clone the repo to your machine.
2. Make sure you have the dependencies installed (`pip install -r requirements.txt`).
3. Run the specific script you need. For example:
```bash
python car_counter.py
```

### 📝 Notes on implementation
* **Masking:** I used custom masks to ignore areas of the video that aren't relevant (like sidewalks in the car counter), which helps reduce false positives.
* **Tracking:** The system assigns a unique ID to each object so it doesn't count the same person or car twice once they've crossed the line.

### To-Do / Future Improvements
* [ ] Add support for multiple camera feeds.
* [ ] Optimize the tracking for low-light video footage.
* [ ] Create a web dashboard to display the count data.

### Why this version works:
* **First-person language:** Using "I built," "I used," and "I used custom masks" makes it clear this is your personal project.
* **No "fluff":** It skips the marketing-style badges and "world-class" adjectives.
* **The "Notes" section:** This shows you actually understand the code (mentioning masks and tracking logic), which is what recruiters or other developers look for.
