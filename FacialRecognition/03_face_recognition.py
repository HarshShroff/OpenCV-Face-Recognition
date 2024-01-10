''''
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc                       
	==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18  

'''

import cv2
import dlib
import numpy as np
from imutils import face_utils
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Load the shape predictor for dlib face landmarks
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

font = cv2.FONT_HERSHEY_SIMPLEX

# initiate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1, etc
names = ['None', 'Harsh'] 

# Initialize and start real-time video capture with Intel RealSense
pipeline = cv2.VideoCapture(1)
while True:
    # Read frames from the camera
    ret, frames = pipeline.read()
    
    # Convert the frame to grayscale for face recognition
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using dlib
    faces = detector(gray)

    for face in faces:
        # Get the facial landmarks
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Draw rectangle around the face
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Predict the face ID
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Check if confidence is less than 100 ==> "0" is a perfect match 
        if confidence < 100:
            id = names[id]
            confidence = " {0}%".format((round(100 - confidence)))
        else:
            id = "unknown"
            confidence = " {0}%".format((round(100 - confidence)))

        # Display the ID and confidence
        cv2.putText(frames, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(frames, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow('Real-Time Face Recognition', frames)

    # Press 'ESC' to exit the video stream
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break

# Clean up
print("\n [INFO] Exiting Program and cleanup stuff")
pipeline.release()
cv2.destroyAllWindows()
