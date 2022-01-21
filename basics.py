import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('images/elon.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('images/elon_test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
EncodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),2)


faceLocTest = face_recognition.face_locations(imgTest)[0]
EncodeElonTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(0,255,0),2)


results = face_recognition.compare_faces([EncodeElon],EncodeElonTest)
distance = face_recognition.face_distance([EncodeElon],EncodeElonTest)
print(results)
print(distance)
cv2.putText(imgTest, f'{results} {round(distance[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)




cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)

 