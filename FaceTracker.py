# --- face_tracking_module.py ---
import cv2
import numpy as np

fbRange = [4000, 6000]  # Default values, can be changed in main script


def findFace(img):
    w,h=360, 240
    faceCascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)
    myFaceListC, myFaceListArea = [], []

    for (x, y, w, h) in faces:
        cx, cy = x + w // 2, y + h // 2
        area = w * h
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    if myFaceListArea:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]


def trackFace(info, w, pid, pError):
    area = info[1]
    x, y = info[0]
    fb = 0

    error = x - w // 2

    # Dead zone
    if abs(error) < 20:
        error = 0

    # PID control
    speed = pid[0] * error + pid[1] * (error - pError)

    # Slow down near center
    if abs(error) < 60:
        speed *= 0.5

    speed = int(np.clip(speed, -30, 30))  # limit yaw speed

    # Forward/Backward logic
    if area > 6000:
        fb = -20
    elif area < 4000 and area != 0:
        fb = 20
    else:
        fb = 0

    if x == 0:
        error = 0
        speed = 0
        fb = 0

    print(f"Forward/Backward: {fb}, Yaw: {speed}, pError: {error}")
    #me.send_rc_control(0, fb, 0, speed)
    return fb, speed, error
