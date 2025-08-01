import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Collected data
data = []
labels = []

# Number of samples per gesture
NUM_SAMPLES = 100

print("Press a key (A, B, C...) to start recording samples for that gesture.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(framergb)
    key = cv2.waitKey(1)

    # Detect hand
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])  # Only x and y

            if key != -1:
                label = chr(key).upper()
                print(f"Recording samples for: {label}")
                for _ in range(NUM_SAMPLES):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands.process(framergb)

                    if result.multi_hand_landmarks:
                        hand_landmarks = result.multi_hand_landmarks[0]
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y])

                        if len(landmarks) == 42:
                            data.append(landmarks)
                            labels.append(label)

                    cv2.putText(frame, f"Collecting {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)
                    cv2.imshow("Frame", frame)
                    cv2.waitKey(30)

    cv2.imshow("Frame", frame)

    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv("hand_gesture_data.csv", index=False)

print("✅ Data collection complete. Saved to hand_gesture_data.csv")
