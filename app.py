import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from sklearn.externals import joblib  # For loading the trained model

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained scikit-learn model
model = joblib.load('emotion_model.pkl')  # Replace with your model path

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# For real-time emotion tracking
emotion_history = deque(maxlen=100)

# Start video capture
cap = cv2.VideoCapture(0)

plt.ion()
fig, ax = plt.subplots()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float') / 255.0

        # Flatten and reshape for the model
        roi_gray = roi_gray.flatten().reshape(1, -1)

        # Predict emotion
        preds = model.predict_proba(roi_gray)[0]
        label = emotion_labels[np.argmax(preds)]

        emotion_history.append(np.argmax(preds))

        # Display results
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Emotion Recognition', frame)

    # Plotting real-time emotion tracking
    if len(emotion_history) > 1:
        ax.clear()
        ax.plot(list(emotion_history))
        ax.set_ylim(0, 6)
        ax.set_yticks(range(len(emotion_labels)))
        ax.set_yticklabels(emotion_labels)
        plt.draw()
        plt.pause(0.001)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()