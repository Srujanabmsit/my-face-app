import face_recognition

def test(image, model_dir, device_id):
    # Your implementation here
    # Example:
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) > 0:
        return 1  # Assuming 1 indicates a real face
    else:
        return 0  # Assuming 0 indicates no face or fake face

