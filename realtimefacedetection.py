import os
import cv2
import time
import numpy as np
from keras.models import model_from_json
from flask import Flask, jsonify, render_template
import threading
from collections import defaultdict

# Flask App Setup
app = Flask(__name__)
emotion_data = defaultdict(float)  # To store emotion durations

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML page

@app.route('/emotion-data')
def get_emotion_data():
    return jsonify(emotion_data)  # Send emotion data as JSON

# Load Emotion Detection Model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load Haar Cascade for Face Detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Helper Function: Preprocess the Image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Start Flask in a Thread
def run_flask():
    app.run(debug=True, use_reloader=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.start()

# Emotion Labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Webcam Emotion Detection
webcam = cv2.VideoCapture(0)
last_emotion = None
last_time = time.time()

def home():
    return "Hello, Render!"
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)


while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        img = extract_features(face)
        pred = model.predict(img)
        current_emotion = labels[pred.argmax()]

        # Update emotion duration
        current_time = time.time()
        if last_emotion is not None:
            emotion_data[last_emotion] += current_time - last_time
        last_emotion = current_emotion
        last_time = current_time

        # Display Emotion on Frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, current_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
