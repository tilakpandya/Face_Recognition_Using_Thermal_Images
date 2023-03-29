import csv
import os
import pandas as pd
import warnings
import cv2
warnings.filterwarnings('ignore')

FILE_NAME = "face_data.csv"
frame_height = frame_width = 0
class dataset:
    def __init__(self):
        self.name = input("Enter Your Name : ")
        self.id = input("Enter Your Id : ")
        self.index = 0
        self.deleted = 0
        self.create_csv()
        self.check()

    def create_csv(self):
        columns = [
                [
                    "Index",
                    "Full_Name",
                    "Id", 
                    "Deleted"
                ]
            ]
        if not os.path.exists(FILE_NAME):    
            with open(FILE_NAME, 'w', newline='') as file:
                # Create a writer object
                writer = csv.writer(file)

                # Write the data to the CSV file
                writer.writerows(columns)

    def add(self):
        df = pd.read_csv(FILE_NAME)
        self.index = df.shape[0] + 1
        row_data = [self.index, self.name, self.id, self.deleted]

        # Open the CSV file in append mode and write the new row of data
        with open(FILE_NAME, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(row_data)
    def check(self):
        df = pd.read_csv(FILE_NAME)
        while True : 
            if int(self.id) in df["Id"].values:
                self.id = input("Entered Id has been used before, please enter your unique Id : ")
            #elif len(self.id) != 9:
                #self.id = input("Entered Id Should be 9 digit number, please enter valid student Id : ")
            else:
                break
    def getName(self, label):
        df = pd.read_csv(FILE_NAME)
        result = df[df["Id"] == int(label)].Full_Name
        return result

def cameraSource(winName):
    url = "rtsp://admin:Admin12345@192.168.1.142/Streaming/channels/2"
    source = cv2.VideoCapture(0)
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    return source

def isImageExist(images, thermal_gray):
    return True
    if(len(images) == 0):
        return True
    for filename in images:
        if np.array_equal(cv2.equalizeHist(thermal_gray), cv2.equalizeHist(image)):
            return False
    return True

def getCordinates(index,detections, frame):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    x_left_bottom = int(detections[0, 0, index, 3] * frame_width)
    y_left_bottom = int(detections[0, 0, index, 4] * frame_height)
    x_right_top = int(detections[0, 0, index, 5] * frame_width)
    y_right_top = int(detections[0, 0, index, 6] * frame_height)

    return x_left_bottom, y_left_bottom, x_right_top, y_right_top

