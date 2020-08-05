# Captures image from webcam video stream
# Extract all faces from the image frame (using haarcascades)
# Stores the face information into numpy arrays

# 1. Read and show stream, capture images
# 2. Detect faces and show bounding box
# 3. Flatten the largest face image (grayscale to save memory) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_alt.xml')

skip = 0

face_data = []
dataset_path = './data/'

file_name = input("Enter the name of the person: ")

while True :

    ret, frame_org = cap.read()
    frame = cv2.flip(frame_org, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not ret :
        continue

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    # print(faces)

    # face_section = cv2.resize(frame, (100, 100))
    
    # sort faces on the basis of area
    faces = sorted(faces, key = lambda f : f[2]*f[3])

    # faces[-1] is the largest face by area
    for face in faces[-1 : ] :
        x, y, w, h = face
        cv2.rectangle(gray_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Extract (crop) rregion of interest (equired part)
        offset = 10 # (pixels) Padding of 10px in all sides
        # first axis is y in openCV
        face_section = gray_frame[y-offset : y+h+offset, x-offset : x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        skip += 1
        # Store every 10th face
        if skip % 10 == 0 :
            face_data.append(face_section)
            print(len(face_data))

        

    # cv2.imshow('frame', frame)
    cv2.imshow('Face Section', face_section)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q') :
        break

# Convert our face list array into numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

# Save this data into file system
np.save(dataset_path + file_name + '.npy', face_data)

print("Data successfully saved at " + dataset_path + file_name + '.npy')

cap.release()
cv2.destroyAllWindows()