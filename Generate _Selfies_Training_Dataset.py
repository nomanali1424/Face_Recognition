import cv2
import numpy as np
import os
import time

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load the haarcascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = [] # for 3rd task
dataset_path = './data/' # for 3rd task data set collection folder

file_name = input("Enter the name of the person (without spaces or special characters): ")
file_name = file_name.strip() # remove leading/trailing white spaces
file_name = file_name.replace(' ', '_') # replace spaces with underscores

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

skip_count = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1
    
    if not ret:
        print("Error: failed to capture video")
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    
    if len(faces) > 0:
        # Pick the first face because it is the largest face
        x, y, w, h = faces[0]
        
        # Creating bounding box
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
        
        # Extract (crop out the required face): region of interest
        offset = 10 
        # Padding of 10 pixels for all directions
        # And the format of pixels are (y,x)
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset] # Slicing
        face_section = cv2.resize(face_section, (100, 100))
        
        # Store every 5th face
        skip_count += 1
        if (skip_count % 5 == 0):
            face_data.append(face_section)
            print(len(face_data))
            
            # Display the face section image
            cv2.imshow("Face section", face_section)
        
    # Display the frame 
    cv2.imshow("Frame", frame)
    
    # Check if the user pressed q to quit
    key_pressed = cv2.waitKey(10) & 0xFF
    if key_pressed == ord('q'):
        print("Exiting the program...")
        break
        
    # Check if the program has collected enough face data
    if len(face_data) >= 100:
        break
    
    # Add a small delay to avoid high CPU usage
    time.sleep(0.01)

# Convert our face list array into a numpy array
if len(face_data) > 0:
    face_data = np.asarray(face_data)
    face_data = face_data.reshape((face_data.shape[0], -1)) # Here no of rows should be faces so [0]
    print(face_data.shape)    # And no of columns can be figured out itself so -1

    # Save this data into file system
    np.save(dataset_path + file_name + '.npy', face_data)
    print("Data Successfully saved at " + dataset_path + file_name + '.npy')
else:
    print("No face data collected")

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
