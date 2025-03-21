# Import necessary libraries
import cv2
import os

# Initialize the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a directory to save the captured images
person_name = 'Person1'  # Name of the person
save_path = f'./faces/{person_name}'
os.makedirs(save_path, exist_ok=True)

# Capture images from the webcam
cap = cv2.VideoCapture(0)
count = 0

print("Press 'q' to stop capturing images.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face and save the face region
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_region = gray[y:y+h, x:x+w]
        file_name = f"{save_path}/{count}.jpg"
        cv2.imwrite(file_name, face_region)
        count += 1

    # Display the frame
    cv2.imshow("Capturing Faces", frame)

    # Press 'q' to stop capturing images
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
