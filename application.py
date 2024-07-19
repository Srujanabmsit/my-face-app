from flask import Flask, render_template, request, jsonify
import os
import cv2
import face_recognition
import pickle
import numpy as np
import base64
import warnings

application = Flask(__name__)

# Load the database of known faces
known_faces_dir = './db'
known_faces = {}
for file in os.listdir(known_faces_dir):
    if file.endswith('.pickle'):
        name = file[:-7]
        with open(os.path.join(known_faces_dir, file), 'rb') as f:
            known_faces[name] = pickle.load(f)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/login', methods=['POST'])
def login():
    data = request.json
    video_stream = data['video_stream']

    # Convert the video stream to an OpenCV image array
    try:
        image_array = np.frombuffer(base64.b64decode(video_stream), np.uint8)
        image_array = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'status': 'failure', 'message': 'Error decoding video stream: {}'.format(str(e))})

    # Recognize the face
    face_locations = face_recognition.face_locations(image_array)
    face_encodings = face_recognition.face_encodings(image_array, face_locations)
    if len(face_encodings) == 0:
        return jsonify({'status': 'failure', 'message': 'No face detected'})
    
    face_encoding = face_encodings[0]
    
    # Compare face encodings with known faces
    face_distances = face_recognition.face_distance(list(known_faces.values()), face_encoding)
    best_match_index = np.argmin(face_distances)
    
    if face_distances[best_match_index] < 0.6:  # Adjust tolerance as needed
        name = list(known_faces.keys())[best_match_index]
        return jsonify({'status': 'success', 'name': name})
    
    return jsonify({'status': 'failure', 'message': 'Unknown user'})

@application.route('/register', methods=['POST'])
def register():
    data = request.json
    video_stream = data.get('video_stream')
    name = data.get('name')

    if not video_stream or not name:
        return jsonify({'status': 'failure', 'message': 'Missing video stream or name'})

    # Convert the video stream to an OpenCV image array
    try:
        image_array = np.frombuffer(base64.b64decode(video_stream), np.uint8)
        image_array = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'status': 'failure', 'message': 'Error decoding video stream: {}'.format(str(e))})

    # Extract the face encoding
    face_locations = face_recognition.face_locations(image_array)
    if len(face_locations) == 0:
        return jsonify({'status': 'failure', 'message': 'No face detected'})
    
    face_encoding = face_recognition.face_encodings(image_array, face_locations)[0]

    # Save the face encoding to the database
    with open(os.path.join(known_faces_dir, '{}.pickle'.format(name)), 'wb') as f:
        pickle.dump(face_encoding, f)

    return jsonify({'status': 'success', 'message': 'User registered successfully'})

warnings.simplefilter("ignore", DeprecationWarning)

if __name__ == '__main__':
    application.run(debug=True)
