import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
# detector = cv2.CascadeClassifier("/Cascades/haarcascade_frontalface_default.xml");
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# function to get the images and label data
def getImagesAndLabels(path):
    faceSamples = []
    ids = []

    for user_folder in os.listdir(path):
        user_folder_path = os.path.join(path, user_folder)
        
        if os.path.isdir(user_folder_path):
            for image_name in os.listdir(user_folder_path):
                image_path = os.path.join(user_folder_path, image_name)
                
                print(f"Processing image: {image_path} for user: {user_folder}")

                if os.path.isfile(image_path):
                    PIL_img = Image.open(image_path).convert('L')
                    img_numpy = np.array(PIL_img, 'uint8')

                    id = int(image_name.split("_")[1])
                    faces = detector.detectMultiScale(img_numpy)

                    for (x, y, w, h) in faces:
                        faceSamples.append(img_numpy[y:y+h, x:x+w])
                        ids.append(id)
                else:
                    print(f"Skipping non-file path: {image_path}")

    print(f"Number of images loaded: {len(faceSamples)}")
    print(f"Number of unique ids: {len(np.unique(ids))}")

    return faceSamples, ids


# Create the 'trainer' directory if it doesn't exist
trainer_dir = 'trainer/'
if not os.path.exists(trainer_dir):
    os.makedirs(trainer_dir)

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
trainer_path = os.path.join(trainer_dir, 'trainer.yml')
recognizer.write(trainer_path)  # recognizer.save() worked on Mac, but not on Pi

# Print the number of faces trained and end the program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
