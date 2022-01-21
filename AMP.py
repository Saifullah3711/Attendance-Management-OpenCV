# Importing all the required libraries
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'imagesAtt'
images = []
classNames = []
myList = os.listdir(path)
print(myList)


# Reading all the images in directory 
# Appending the images and their names to images and classNames
for cl in myList:
    currentImg = cv2.imread(f'{path}/{cl}')
    images.append(currentImg)
    # os.path.splittext -> split into basename and extension
    classNames.append(os.path.splitext(cl)[0])
    
def findEncodings(images):
    listEncodings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(img)[0]
        listEncodings.append(encoding)
    return listEncodings 

# Check if the name is in csv, don't add
# it not available, add it to the csv
def attendanceMark(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            nowTime = datetime.now()
            dtString = nowTime.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Completed .......')


cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrent = face_recognition.face_locations(imgS)
    facesEncodings = face_recognition.face_encodings(imgS, facesCurrent)

    for faceLoc, faceEnc in zip(facesCurrent, facesEncodings):
        # compare_faces compare test face with list of known faces
        matches = face_recognition.compare_faces(encodeListKnown,faceEnc)
        faceDist = face_recognition.face_distance(encodeListKnown,faceEnc)
        
        matchedIndex = np.argmin(faceDist)
        
        if matches[matchedIndex]:
            name = classNames[matchedIndex].upper()
            y1,x2,y2,x1 = faceLoc

            
            # images were resized by 0.25 
            # Rescaling back for the original imags 
            y1,x2,y2,x1 = 4*y1,4*x2,4*y2,4*x1
            cv2.rectangle(img, (x1, y1), (x2,y2),(0,255,0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2,y2),(0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2)
            attendanceMark(name)

    cv2.imshow("image", img)
    cv2.waitKey(1)

