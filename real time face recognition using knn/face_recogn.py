# Haarcascade is a pre-trained classifier to identify facial features
# Old method. New method is using CNN
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


while True :
    ret, frame_org = cap.read()
    frame = cv2.flip(frame_org, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not ret :
        continue
    faces = face_cascade.detectMultiScale(gray_frame, 1.2, 5) # 5 is no. of neighboures
    # Scale factor specifies how much image size is reduced at each scale
    # Each image will shrink by 30% on each pass
    # Higher values means less detections but with high accuracy
    # Returns tuples of : coordinates of top left corner of detected faces and height and width
    
    for (x, y, w, h) in faces :
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
        # 2nd parameter is opposite end points of rectangle
    
    cv2.imshow('Video Frame', frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()