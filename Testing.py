import cv2
import numpy as np
import os
from sklearn.svm import SVC


# Function to calculate distance between two vectors
def distance(v1, v2):
    return np.sqrt(((v1-v2)**2).sum())


# Function to train SVM classifier
def train_svm(train_data, train_labels):
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(train_data, train_labels.ravel())
    return svm


# Function to predict label using SVM classifier
def predict(test_data, svm):
    return svm.predict(test_data.reshape(1, -1))


# Init Camera
cap = cv2.VideoCapture(0)

# Load face detection model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# Set up variables for face data and labels
skip = 0
dataset_path = './data/'
face_data = []
labels = []
class_id = 0 # Label for given file
names = {} # Mapping between id - name

# Data Preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        # Create a mapping between class_id and name
        names[class_id] = fx[:-4]
        print("Loaded " + fx)
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        # Create labels for the class
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

# Concatenate face data and labels
face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

print(face_dataset.shape)
print(face_labels.shape)

# Concatenate face dataset and labels
trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

# Train SVM classifier
svm = train_svm(trainset[:, :-1], trainset[:, -1])

# Testing
while True:
    ret, frame = cap.read()
    if ret == False:
        continue

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for face in faces:
        x, y, w, h = face

        # Get the face ROI
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]

        if not face_section.size:
            continue

        face_section = cv2.resize(face_section, (100, 100))

        # Predict label using SVM classifier
        out = predict(face_section.flatten(), svm)

        # Display the predicted name and rectangle around the face
        pred_name = names[int(out)]
        cv2.putText(frame, pred_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow("Faces", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
