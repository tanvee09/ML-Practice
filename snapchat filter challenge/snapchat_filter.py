import cv2

# frame = cv2.imread('./test/Before.png')
frame = cv2.imread('Jamie_Before.jpg')
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
# cv2.rectangle(frame, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0, 0, 255), 5)
glasses = cv2.resize(glasses, (eye_w, eye_h))
print(glasses.shape, eye_w, eye_h, eye_x, eye_y)
for i in range (eye_w) :
    for j in range (eye_h) :
        if glasses[j, i, 3] != 0 :
            frame[j + eye_y, i + eye_x, :] = glasses[j, i, :3]
# cv2.imshow('Glasses', glasses)

nose_x, nose_y, nose_w, nose_h = nose[-1]
cv2.rectangle(frame, (nose_x, nose_y), (nose_x + nose_w, nose_y + nose_h), (255, 0, 0), 5)
mustache = cv2.resize(mustache, (nose_w, nose_h))
print(mustache.shape, nose_w, nose_h, nose_x, nose_y)
for i in range (nose_w) :
    for j in range (nose_h) :
        if mustache[j, i, 3] != 0 :
            frame[j + nose_y, i + nose_x, :] = mustache[j, i, :3]
# cv2.imshow('Mustache', mustache)

cv2.imshow('Eyes and Nose', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()