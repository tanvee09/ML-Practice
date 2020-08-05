# Use KNN to classify face

# 1. Load the training data (numpy arrays of all persons)
#     x-values are stored in the numpy arrays
#     y-values we need to assign for each person
# 2. Read a video using openCV
# 3. Extract faces out of it
# 4. Use KNN to find the prediction of face (int)
# 5. Map the predicted id to the name of the user
# 6. Display the predictions on the screen - bounding box and name

import numpy as np
import cv2
import os

def dist(x1, x2) :
    return np.sqrt(((x1-x2)**2).sum())

def knn(train, test, k = 5) :
    vals = []
    m = train.shape[0] # Total points
    for i in range(m) :
        ix = train[i, :-1]
        iy = train[i, -1]
        d = dist(test, ix)
        vals.append((d, iy))
    vals = sorted(vals)
    # Nearest/First k points
    vals = vals[:k]
    vals = np.array(vals)[:, -1]
    # print(vals)
    output = np.unique(vals, return_counts = True)
    # print(new_vals)
    max_freq_index = output[1].argmax()
    return output[0][max_freq_index]

# Initialise camera

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_alt.xml')

skip = 0

dataset_path = './data/'

face_data = [] # Training data
labels = []

class_id = 0
names = {} # Mapping between id and name


# Data preparation
for fx in os.listdir(dataset_path) : # listdir (dir in windows) gives all the files in the path
    if fx.endswith('.npy') :
        names[class_id] = fx[:-4]
        print("Loaded " + fx)
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item) # face_data -> [[X1], [X2], ..., [Xm]]

        # data_item is np array of dimensions -> 10 faces -> (10, 10000)
        # for class_id = 0, taget will be array of 10 zeros
        # Create labels for the class
        target = class_id * np.ones((data_item.shape[0], ))
        class_id += 1
        labels.append(target) # labels -> [[Y1], [Y2], ..., [Ym]]

face_dataset = np.concatenate(face_data, axis = 0)
face_labels = (np.concatenate(labels, axis = 0)).reshape((-1, 1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis = 1)
print(trainset.shape) # 10000 features and 1 label

# Testing
while True :
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not ret :
        continue
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for face in faces :
        x, y, w, h = face

        # Get the face ROI (Region of Interest)'
        offset = 10
        face_section = gray_frame[y-offset : y+h+offset, x-offset : x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))
        # Predicted label
        output = knn(trainset, face_section.flatten())
        pred_name = names[int(output)]
        # Display name and rectangle around face
        cv2.putText(gray_frame, pred_name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.rectangle(gray_frame, (x,y), (x+w,y+h), (255,255,0), 5)

    cv2.imshow("Faces", gray_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()