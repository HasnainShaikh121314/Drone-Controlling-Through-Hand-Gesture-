import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from djitellopy import tello
from time import sleep
import cv2



 

# Load the trained model and label encoder
model = load_model("hand_gesture_model_new.h5")
with open("label_encoder_new.pkl", "rb") as f:
    le = pickle.load(f)

# Gesture label mapping (Old label -> New label)
label_map = {
    'A': 'Stop',
    'B': 'Up',
    'C': 'Down',
    'D':'left',
    'E':'right',
    'F':'flip',
    'G':'turn left',
    'H':'turn right',
    'I':'move backward',
    'J':'move farward'
    
}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    #frame=me.get_frame_read().frame
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame for a mirror effect
    h, w, _ = frame.shape
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            # Reshape landmarks to (1, 42) to match the model input
            sample_landmarks = np.array(landmarks).reshape(1, -1)

            # Predict using the model
            prediction = model.predict(sample_landmarks)
            predicted_class = np.argmax(prediction)
            predicted_gesture_old = le.inverse_transform([predicted_class])[0]

            # Map the old label to the new label (e.g., A -> Stop)
            predicted_gesture_new = label_map.get(predicted_gesture_old, predicted_gesture_old)

            # Show the predicted gesture on the frame
            cv2.putText(frame, f"Gesture: {predicted_gesture_new}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with the prediction
    cv2.imshow("Frame", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
