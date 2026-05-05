import cv2
import face_recognition
import os
import csv
from datetime import datetime

path = 'dataset'
images = []
classNames = []

for file in os.listdir(path):
    img = cv2.imread(f'{path}/{file}')
    images.append(img)
    classNames.append(os.path.splitext(file)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('attendance.csv', 'a+', newline='') as f:
        writer = csv.writer(f)
        now = datetime.now()
        timeString = now.strftime('%H:%M:%S')
        writer.writerow([name, timeString])

print("Encoding faces...")
encodeListKnown = findEncodings(images)
print("Encoding complete")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    imgSmall = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = faceDis.argmin()

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, name, (x1,y2+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            markAttendance(name)

    cv2.imshow('Face Recognition Attendance', img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
