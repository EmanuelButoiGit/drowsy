import cv2
import winsound


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
duration = 1000  # milliseconds
freq = 440  # Hz
cv2.namedWindow("Webcam")
cap = cv2.VideoCapture(0)

if cap.isOpened(): # try to get the first frame
    #_, img = cap.read()
    rval, frame = cap.read()
else:
    rval = False

while rval:
    detected = False
    detected_face = False

    cv2.imshow("Webcam", frame)
    rval, frame = cap.read()

    _, img = cap.read()
    rval, frame = cap.read()

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # viteza de redare 1.2
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)  # 4=vecini (mean)

    for (x, y, width, height) in faces:
        detected_face = True
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 2)

    roi_gray = gray[y:y + height, x:x + width]
    roi_color = img[y:y + height, x:x + width]

    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        detected=True
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    if(detected==False and detected_face==True):
        winsound.Beep(freq, duration)
        cv2.putText(img, "Don't sleep while driving!!!", (x-250 , y - 200), cv2.FONT_ITALIC, 1.5, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Face detection', img)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("Webcam")








