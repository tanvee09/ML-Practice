import cv2
import pandas as pd

cap = cv2.VideoCapture(0)

eyes_cascade = cv2.CascadeClassifier('./third-party/frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('./third-party/Nose18x15.xml')

must = cv2.imread('mustache.png', cv2.IMREAD_UNCHANGED)
glass = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED)


while True :
    ret, frame_org = cap.read()
    frame = cv2.flip(frame_org, 1)

    if not ret :
        continue
    eyes = eyes_cascade.detectMultiScale(frame, 1.2, 5)
    nose = nose_cascade.detectMultiScale(frame, 1.2, 5)

    # eyes = sorted(eyes, key = lambda f : f[2]*f[3])
    # nose = sorted(nose, key = lambda f : f[2]*f[3])
        
    if len(eyes) > 0 :
        eye_x, eye_y, eye_w, eye_h = eyes[-1]
        # cv2.rectangle(frame, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0, 0, 255), 2)
        glasses = cv2.resize(glass, (eye_w + 16, eye_h + 30))
        # print(glasses.shape, eye_w, eye_h, eye_x, eye_y)
        for i in range (eye_w + 16) :
            for j in range (eye_h + 30) :
                if glasses[j, i, 3] != 0 :
                    frame[j + eye_y - 10, i + eye_x - 9, :] = glasses[j, i, :3]
        # cv2.imshow('Glasses', glasses)

    if len(nose) > 0 :
        nose_x, nose_y, nose_w, nose_h = nose[-1]
        # cv2.rectangle(frame, (nose_x, nose_y), (nose_x + nose_w, nose_y + nose_h), (255, 0, 0), 2)
        mustache = cv2.resize(must, (nose_w + 10, nose_h - 5))
        # print(mustache.shape, nose_w, nose_h, nose_x, nose_y)
        for i in range (nose_w + 10) :
            for j in range (nose_h - 5) :
                if mustache[j, i, 3] != 0 :
                    frame[j + nose_y + 25, i + nose_x, :] = mustache[j, i, :3]
        
    cv2.imshow('Video Frame', frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()