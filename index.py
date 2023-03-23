import cv2
import os
import helper
import numpy as np
from PIL import ImageChops

dataset = helper.dataset()
dataset.add()

# Getting last file index
number =1
count = 0
imagesList = []
win_name = 'Camera Preview'
source = helper.cameraSource(win_name)


# Create a directory to save the dataset
if not os.path.exists("dataset"):
    os.makedirs("dataset")

net = cv2.dnn.readNetFromCaffe("deploy.prototxt","res10_300x300_ssd_iter_140000_fp16.caffemodel")

# 27 is ESC key value
while cv2.waitKey(1) != 27:
    if count > 100:
        break
    has_frame, frame = source.read()
    if has_frame is True:
        # To flip the video image
        frame = cv2.flip(frame,1)
        # Create a 4D blob from a frame. it converts image to numeric multidimentional array.
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB = False, crop = False)
        # Run a model
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            #check the threshold of confidence
            if confidence < 0.7:
                continue

            x_left_bottom, y_left_bottom, x_right_top, y_right_top = helper.getCordinates(i,detections, frame)

            # A rectangle for the face
            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))

            label_size, base_line = cv2.getTextSize("Confidence: %.4f" % confidence, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # A rectangle for showing confidence percentage
            cv2.rectangle(frame, (x_left_bottom, y_left_bottom - label_size[1]),
                                (x_left_bottom + label_size[0], y_left_bottom + base_line),
                                (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, "Confidence: %.4f" % confidence, (x_left_bottom, y_left_bottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            
            # Convert the face image to a thermal image using cv2.applyColorMap
            face_image = frame[y_left_bottom:y_right_top, x_left_bottom:x_right_top]
            if face_image.size != 0:
                thermal_image = cv2.applyColorMap(cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
            else:
                continue        
            rgb_image = frame[y_left_bottom:y_right_top, x_left_bottom:x_right_top] 

            # Resize the images to the same size
            rgb_image = cv2.resize(rgb_image, thermal_image.shape[:2][::-1])

            # Convert the thermal image to grayscale and apply a color map
            thermal_gray = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
            thermal_color = cv2.applyColorMap(thermal_gray, cv2.COLORMAP_JET)
        
            if(helper.isImageExist(imagesList,thermal_gray)):
                imageName = "fused_face_" + str(dataset.id) + "_" + str(count) + ".jpg"
                imagesList.append(imageName)
                # Blend the RGB and thermal images
                blended = cv2.addWeighted(rgb_image, 0.5, thermal_color, 0.5, 0)
                # Save the blended image
                cv2.imwrite("dataset/"+imageName, blended)
                count=count+1

        cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)
