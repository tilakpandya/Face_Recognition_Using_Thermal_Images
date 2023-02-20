import csv
import os
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')

FILE_NAME = "face_data.csv"
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
            else:
                break
    def getName(self, label):
        df = pd.read_csv(FILE_NAME)
        result = df[df["Id"] == int(label)].Full_Name
        return result     

# Testing ClassNa
sd = dataset()
print(sd.getName(200535046))
#sd.add()

#class register : 
#    def __init__(self):
#        self.name = input("Enter your name :")
#        self.id = input("Enter your Id :")
        
#    def saveImages(self):

