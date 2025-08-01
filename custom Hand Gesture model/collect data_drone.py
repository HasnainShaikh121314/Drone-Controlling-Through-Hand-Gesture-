import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from djitellopy import tello
from time import sleep
import cv2

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Webcam


# Collected data
data = []
labels = []

# Number of samples per gesture
NUM_SAMPLES = 100

print("Press a key (A, B, C...) to start recording samples for that gesture.")

me =tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()
while True:
    img=me.get_frame_read().frame
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
                    img=me.get_frame_read().frame
                    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


cv2.destroyAllWindows()

# Save to CSV
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv("hqand_gesture_data.csv", index=False)

print("✅ Data collection complete. Saved to hand_gesture_data.csv")
