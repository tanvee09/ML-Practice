import cv2
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

frame = cv2.imread('./test/Before.png')
# frame = cv2.imread('Jamie_Before.jpg')
# cv2.imshow('Image', frame)

eyes_cascade = cv2.CascadeClassifier('./third-party/frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('./third-party/Nose18x15.xml')

mustache = cv2.imread('mustache.png', cv2.IMREAD_UNCHANGED)
glasses = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED)

eyes = eyes_cascade.detectMultiScale(frame, 1.2, 5)
nose = nose_cascade.detectMultiScale(frame, 1.2, 5)

eyes = sorted(eyes, key = lambda f : f[2]*f[3])
nose = sorted(nose, key = lambda f : f[2]*f[3])
    
eye_x, eye_y, eye_w, eye_h = eyes[-1]
# cv2.rectangle(frame, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0, 0, 255), 2)
glasses = cv2.resize(glasses, (eye_w + 16, eye_h + 30))
print(glasses.shape, eye_w, eye_h, eye_x, eye_y)
for i in range (eye_w + 16) :
    for j in range (eye_h + 30) :
        if glasses[j, i, 3] != 0 :
            frame[j + eye_y - 10, i + eye_x - 9, :] = glasses[j, i, :3]
# cv2.imshow('Glasses', glasses)

nose_x, nose_y, nose_w, nose_h = nose[-1]
# cv2.rectangle(frame, (nose_x, nose_y), (nose_x + nose_w, nose_y + nose_h), (255, 0, 0), 2)
mustache = cv2.resize(mustache, (nose_w + 10, nose_h - 5))
print(mustache.shape, nose_w, nose_h, nose_x, nose_y)
for i in range (nose_w + 10) :
    for j in range (nose_h - 5) :
        if mustache[j, i, 3] != 0 :
            frame[j + nose_y + 25, i + nose_x, :] = mustache[j, i, :3]
# cv2.imshow('Mustache', mustache)

cv2.imshow('Eyes and Nose', frame)
# cv2.imwrite('answer.png', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()

print(frame.shape)
frame = frame.reshape((-1, 3))
print(frame.shape)

df = pd.DataFrame(frame, dtype = 'uint8', columns = ['Channel 1', 'Channel 2', 'Channel 3'])
df.to_csv('ans_csv.csv', index = False)

# df2 = pd.read_csv('ans_csv.csv')
# data = df2.values
# # cv2.imshow('blabla', data.reshape(485, 377, 3))
# cv2.waitKey(0)
# cv2.destroyAllWindows()