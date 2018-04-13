import cv2

face_cascade = cv2.CascadeClassifier('./xml/haarcascade-frontalface-default.xml')
mouth_cascade = cv2.CascadeClassifier('./xml/haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier('./xml/haarcascade-eye.xml')

def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#convert rgb to gray
    #detect face in frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)#"5" is a value for the sensitivity. if you want to increase the sensitivity decrease the "5". But this can also cause you to choose the wrong places as a face. Try to find the best value for you.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)#draw rectangle
        gray_face = gray[y:y + h, x:x + w]#chose face cordinants in gray frame
        color_face = frame[y:y + h, x:x + w]#chose face cordinants in color frame
        #detect eys in face
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 15)#"15" is a value for the sensitivity. if you want to increase the sensitivity decrease the "15". But this can also cause you to choose the wrong places as a face. Try to find the best value for you.
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(color_face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)#draw rectangle
        #detect mouth in face
        mouth = mouth_cascade.detectMultiScale(gray_face, 1.1, 20)#"20" is a value for the sensitivity. if you want to increase the sensitivity decrease the "20". But this can also cause you to choose the wrong places as a face. Try to find the best value for you.
        for (sx, sy, sw, sh) in mouth:
            cv2.rectangle(color_face, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)#draw rectangle
    return frame

# cap = cv2.VideoCapture("videotest.mp4") #if tou want use video use this line and delete make the following line comment line
cap = cv2.VideoCapture(0)#if you use one more camera, change "0". 0 is a webcam, "1" second camera and "2" third ...

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = detect(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()