import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from playsound import playsound
import time
import threading
import streamlit as st

# ============================== #
# Configuration and Parameters  #
# ============================== #

EAR_THRESH = 0.2
EAR_CONSEC_FRAMES = 15
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))
ALERT_SOUND_PATH = "D:/okDriver/beeps1.wav"
SHAPE_PREDICTOR_PATH = "C:\\Desktop\\Projects\\python projects\\object_detection\\minor\\shape_predictor_68_face_landmarks_1.dat"
HAAR_CASCADE_PATH = "C:\\Desktop\\Projects\\python projects\\object_detection\\minor\\haarcascade_frontalface_default.xml"

# ============================== #
# Utility Functions             #
# ============================== #

def play_alert_sound():
    threading.Thread(target=playsound, args=(ALERT_SOUND_PATH,), daemon=True).start()

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_eye_landmarks(shape):
    left_eye = [shape[i] for i in LEFT_EYE]
    right_eye = [shape[i] for i in RIGHT_EYE]
    return left_eye, right_eye

def draw_eye_landmarks(frame, eyes):
    for (x, y) in eyes:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

# ============================== #
# EAR-based Drowsiness Check    #
# ============================== #

def check_ear_drowsiness(frame, gray, detector, predictor, state):
    faces = detector(gray)
    for face in faces:
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        left_eye, right_eye = get_eye_landmarks(shape)
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        state['ear_values'].append(ear)
        if len(state['ear_values']) > 10:
            state['ear_values'].pop(0)

        smoothed_ear = np.mean(state['ear_values'])
        dynamic_thresh = max(EAR_THRESH, smoothed_ear * 0.8)

        draw_eye_landmarks(frame, left_eye + right_eye)
        print(f"Raw EAR: {ear:.2f}, Smoothed EAR: {smoothed_ear:.2f}, Dynamic Threshold: {dynamic_thresh:.2f}")

        if smoothed_ear < dynamic_thresh:
            if state['drowsy_start_time'] is None:
                state['drowsy_start_time'] = time.time()
            else:
                elapsed = time.time() - state['drowsy_start_time']
                if elapsed > 1 and not state['alert_played']:
                    play_alert_sound()
                    state['alert_played'] = True

            state['counter'] += 1
            if state['counter'] >= EAR_CONSEC_FRAMES:
                cv2.putText(frame, "EAR DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            state['counter'] = 0
            state['drowsy_start_time'] = None
            state['alert_played'] = False

# ============================== #
# Head Position Detection       #
# ============================== #

def check_head_position_drowsiness(frame, gray, state, face_cascade):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if state['initial_y'] is None:
            state['initial_y'] = y

        vertical_shift = y - state['initial_y']
        if vertical_shift > 10:
            if state['head_drowsy_start'] is None:
                state['head_drowsy_start'] = time.time()
            else:
                elapsed = time.time() - state['head_drowsy_start']
                if elapsed > 2:
                    cv2.putText(frame, "HEAD POSITION ALERT!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if not state['head_alert_played']:
                        play_alert_sound()
                        state['head_alert_played'] = True
        else:
            state['head_drowsy_start'] = None
            state['head_alert_played'] = False

# ============================== #
# Main Video Loop               #
# ============================== #

def main():
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    state = {
        'ear_values': [],
        'counter': 0,
        'drowsy_start_time': None,
        'alert_played': False,
        'initial_y': None,
        'head_drowsy_start': None,
        'head_alert_played': False
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        check_ear_drowsiness(frame, gray, detector, predictor, state)
        check_head_position_drowsiness(frame, gray, state, face_cascade)

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# ============================== #
# Streamlit Interface           #
# ============================== #

def streamlit_app():
    st.title("🚗 Driver Drowsiness Detection System")
    st.markdown("This app uses **Eye Aspect Ratio (EAR)** and **Head Position** to detect drowsiness in real-time via webcam.")
    
    st.warning("⚠️ Webcam window will open on clicking the button. Press ESC to exit the webcam.")

    if st.button("Start Drowsiness Detection"):
        main()

if __name__ == "__main__":
    streamlit_app()
