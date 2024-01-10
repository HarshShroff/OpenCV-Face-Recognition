import cv2
import os

# Set the dataset directory
dataset_dir = "dataset/"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Use the Haarcascades XML file from OpenCV data
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
while True:
    try:
        face_id = int(input('\nEnter user id and press <return>: '))
        name = str(input('\nEnter user name and press <return>: '))
        if face_id > 0:
            break
        else:
            print("Invalid user id. Please enter a positive integer.")
    except ValueError:
        print("Invalid input. Please enter a valid integer.")

# Create a user-specific folder if it doesn't exist
user_folder = os.path.join(dataset_dir, name)
if not os.path.exists(user_folder):
    os.makedirs(user_folder)

print(f"\n[INFO] Initializing face capture for user '{name}'. Look at the camera and wait...")

# Initialize individual sampling face count
count = 0

cam = cv2.VideoCapture(1)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the user-specific folder
        cv2.imwrite(f"{user_folder}/User_{face_id}_{name}_{count}.jpg", gray[y:y+h, x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27 or count >= 100:  # Take 30 face samples or press 'ESC' to stop
        break

# Do cleanup
print("\n[INFO] Exiting Program and cleaning up")
cam.release()
cv2.destroyAllWindows()
