from djitellopy import tello
#from FaceTracker import findFace, trackFace
import KeyPressModule as kp
from GestureDetector import PredictClass
import time
import numpy as np
import cv2

kp.init()

me = tello.Tello()

me.connect()

me.streamon()
label_map = {
    'A': 'stop',
    'B': 'up',
    'C': 'down',
    'D':'left',
    'E':'right',
    'F':'flip',
    'G':'turn left',
    'H':'turn right',
    'I':'move backward',
    'J':'move farward'}

fmode=False
def gestureControl(gesture,fmode):
    
    lr, fb1, ud, yv = 0, 0, 0, 0
    speed1 = 30
    first=0
    if kp.getKey("LEFT") or gesture == "right":
        lr = -speed1

    elif kp.getKey("RIGHT") or gesture == "left":
        lr = speed1

    if kp.getKey("UP")or gesture == "move farward":
        fb1 = speed1

    elif kp.getKey("DOWN")or gesture == "move backward":
        fb1 = -speed1

    if  kp.getKey("w") or gesture == "up":
        ud = speed1

    elif kp.getKey("s")or gesture == "down" :#s
        ud = -speed1
    
    if kp.getKey("a") or gesture == "turn right":
        yv = -speed1

    elif kp.getKey("d") or gesture == "turn left":
        yv = speed1

    if  kp.getKey("q") or gesture == "stop":
        me.land(); time.sleep(3)
    if gesture == "flip":
       fmode = not fmode 
        
        
        #me.flip("f")
   # if first==0:
    if kp.getKey("e"): me.takeoff()
       # first=1

    
    return [lr, fb1, ud, yv,fmode]

        
  
    
gesture1="None"
pError = 0
gesture_u=0

gesture_s=0
gesture_d=0
gesture_l=0

gesture_r=0
gesture_tr=0
gesture_tl=0
gesture_f=0

gesture_mf=0
gesture_mb=0
gesture_f=0

start_point = (0,0)  # x1, y1
end_point = (130, 130)   # (x, y)

# Define the color in BGR (Blue, Green, Red)
color = (0, 255, 0)  # Green

# Define the thickness (use -1 to fill the rectangle)
thickness = 2
w, h = 360, 240

fbRange = [4000, 6000]

pid = [0.2, 0.2, 0]


def findFace(img):
    faceCascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    if len(faces) == 0:
        return img, [[0, 0], 0]

    # Select the biggest face
    biggest = None
    maxArea = 0
    for (x, y, w, h) in faces:
        area = w * h
        if area > maxArea:
            biggest = (x, y, w, h)
            maxArea = area

    if biggest is not None:
        x, y, w, h = biggest
        cx = x + w // 2
        cy = y + h // 2
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return img, [[cx, cy], maxArea]
    
    return img, [[0, 0], 0]
    
def trackFace( info, w, pid, pError ):

    area = info[1]

    x, y = info[0]

    fb = 0

    error = x - w // 2

    speed = pid[0] * error + pid[1] * (error - (pError if pError is not None else 0))


    speed = int(np.clip(speed*1.5, -100, 100))

    if area > fbRange[0] and area < fbRange[1]:

        fb = 0

    elif area > fbRange[1]:

        fb = -20

    elif area < fbRange[0] and area != 0:

        fb = 20

    if x == 0:

        speed = 0

        error = 0

    #print(speed, fb)
    print("Fb",fb,"ud",0,"yaw",speed)
    me.send_rc_control(0, fb, 0, speed)
# Main loop
#cap = cv2.VideoCapture(0)
while True:
    
    img = me.get_frame_read().frame
  #  ret, img = cap.read()
   # if not ret:
    #    break
    img = cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    x1, y1, x2, y2 = 0, 0, 130, 130
    
    fr = img[y1:y2, x1:x2]
    fr1=cv2.resize(fr, (600,600 ))
   
    gesture, processed_frame = PredictClass(fr1)
    

    if gesture=="move farward" or gesture=="flip" or gesture=="up" or gesture=="stop" or gesture=="down" or gesture=="left" or gesture=="right" or gesture=="turn left" or gesture=="turn right" or gesture=="flip" or gesture=="move backward" :
        if gesture=="up":
            gesture_u+=1
            gesture_mf
            gesture_d=0
            gesture_tr=0
            gesture_mb=0
            gesture_tl=0
            gesture_l=0
            gesture_r=0
            if gesture_u==5 and gesture_u <=10:
                gesture1=gesture
                if gesture_u>=10:
                    gesture_u=0
        if gesture=="flip":
            gesture_f+=1
            if gesture_f==20:
                gesture1=gesture
                gesture_f=0
        if gesture=="stop":
            gesture_s+=1
            if gesture_s==20:
                gesture1=gesture
                gesture_s=0
                 
        if gesture=="down" :
            gesture_d+=1
            if gesture_d>=5 and gesture_d<=10:
                
                gesture1=gesture
                if gesture_d>=10:
                    gesture_d=0
        if gesture=="move farward":
            gesture_mf+=1
            gesture_u=0
            gesture_d=0
            gesture_tr=0
            gesture_mb=0
            gesture_tl=0
            gesture_l=0
            gesture_r=0
            if gesture_mf>=5 and gesture_f<=10:
                gesture1=gesture
                if gesture_mf>=10:
                   gesture_mf=0
        if gesture=="move backward":
            gesture_mb+=1
            gesture_u=0
            gesture_d=0
            gesture_tr=0
            gesture_mf=0
            gesture_tl=0
            gesture_l=0
            gesture_r=0
            if gesture_mb>=5 and gesture_mb<=10:
                gesture1=gesture
                if gesture_mb>=10:
                    
                    gesture_mb=0
                
        if gesture=="left":
            
            gesture_l+=1
            gesture_mb=0
            gesture_u=0
            gesture_d=0
            gesture_tr=0
            gesture_mf=0
            gesture_tl=0
            gesture_r=0
            if gesture_l==5 or gesture_l<=10:
                
                gesture1=gesture
                if gesture_l >=10:
                    gesture_l=0
                
        if gesture=="right":
            gesture_r+=1
            gesture_l=0
            gesture_mb=0
            gesture_u=0
            gesture_d=0
            gesture_tr=0
            gesture_mf=0
            gesture_tl=0
            if gesture_r==5 or gesture_r<=10:
                gesture1=gesture
                if gesture_r>=10:
                    
                    gesture_r=0
                
        if gesture=="turn left":
            gesture_tl+=1
            gesture_d=0
            gesture_tr=0
            gesture_mf=0
            gesture_mb=0
            gesture_u=0
            gesture_l=0
            gesture_r=0
            if gesture_tl>=5 and gesture_tl<=10:
                
                gesture1=gesture
                if gesture_tl==10:
                
                    gesture_tl=0
        if gesture=="turn right":
            gesture_tr+=1
            gesture_d=0
            gesture_u=0
            gesture_l=0
            gesture_r=0
            gesture_tl=0
            gesture_mf=0
            gesture_mb=0
            if gesture_tr>=5 and gesture_tr<=10:
                gesture1=gesture
                if gesture_tr>=10:
                    gesture_tr=0
    vals = gestureControl(gesture1,fmode)
  #  vals = gestureControl(gesture1, fmode)
    fmode = vals[4] 
    #fmode=True
    if fmode==False:
        me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
        print(vals[0], vals[1], vals[2], vals[3])
        
    if fmode ==True:
        
        img2 = cv2.resize(img, (w, h))

        img, info = findFace(img2)

        pError = trackFace( info, w, pid, pError)
        
   #    me.send_rc_control(0, fb, 0, yaw)
# Draw the rectangle on the image
    mode_text = "Face Tracking Mode" if fmode else "Gesture Mode"
    cv2.putText(img, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    charge=me.get_battery()
    cv2.putText(img, str(charge), (220, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    img=cv2.rectangle(img, start_point, end_point, color, thickness)
    cv2.imshow("part  View", processed_frame)
    cv2.imshow("Drone View", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    gesture1="None"
    
#me.land()
#cap.release()
cv2.destroyAllWindows()
