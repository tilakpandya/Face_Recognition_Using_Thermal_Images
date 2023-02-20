import cv2
import os
import random
import sys
import numpy as np
import pandas as pd
s = 0
count = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

# Create a directory to save the dataset
if not os.path.exists("dataset"):
    os.makedirs("dataset")

net = cv2.dnn.readNetFromCaffe("deploy.prototxt","res10_300x300_ssd_iter_140000_fp16.caffemodel")

# Getting last file index

folder_path = 'dataset' # replace with the path to the folder you want to check
face_images = []
face_labels = []
number =1
df = pd.read_csv("face_data.csv")
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        image = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        face_images.append(image)
        names = filename.split("_")
        if "thermal" in names:
            print(names)
            name = int(names[2])
        else:
            print(names)
            name = int(names[1])
        face_label = name
        face_labels.append(face_label)
        
# Create the face recognition model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_images, np.array(face_labels))

source = cv2.VideoCapture(s)
win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

if os.path.isdir(folder_path):
    files = os.listdir(folder_path)
    if files:
        print(str(files[len(files)-1]))
        parts = files[len(files)-1].split("_")

        # extract the number from the second element
        number += int(parts[2])

# 27 is ESC key value
while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    
    # To flip the video image
    frame = cv2.flip(frame,1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Detect faces in the current frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB = False, crop = False)
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < 0.7:
            continue
        x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
        y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
        x_right_top = int(detections[0, 0, i, 5] * frame_width)
        y_right_top = int(detections[0, 0, i, 6] * frame_height)

        # Extract the face region from the current frame
        face_image = cv2.cvtColor(frame[y_left_bottom:y_right_top, x_left_bottom:x_right_top], cv2.COLOR_BGR2GRAY)

        # Recognize the face using the face recognition model
        label, confidence = face_recognizer.predict(face_image)

        label_text = ""
        if(confidence > 50):
            label_text = "Unknown"
        else:
            result = df[df["Id"] == int(label)]["Full_Name"]
            label_text = str(result[0]) if result.shape[0] > 0 else "Unknown"
        

        # Draw a rectangle around the face and label it with the recognized name
        #label_text = "Unknown" if confidence > 50 else "Person " + str(label)
        cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
        cv2.putText(frame, label_text, (x_left_bottom, y_left_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)
