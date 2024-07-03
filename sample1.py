import cv2
import mediapipe as mp
from datetime import datetime
import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize MediaPipe Face Detection, Hands, and Pose solutions
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize video capture
video = cv2.VideoCapture(0)

# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='jpg', help="Image extension")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="Output video file name")
args = vars(ap.parse_args())

# Create directories for storing detected activity images
if not os.path.exists('unusual_activities (danger)'):
    os.makedirs('unusual_activities (danger)')

if not os.path.exists('normal_activities'):
    os.makedirs('normal_activities')

if not os.path.exists('suspicious_activities'):
    os.makedirs('suspicious_activities')

# Load pre-trained facial expression recognition model
model = load_model('C:/Users/vedul/Downloads/emotion_detection_model.h5')  # Replace with the path to your model

# Define emotion labels (assuming the model predicts these labels)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Threshold for unusual activity
threshold_distance = 50  # Adjust this value as needed

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def is_angry_face(face_bbox, frame):
    x, y, w, h = int(face_bbox.xmin * frame_width), int(face_bbox.ymin * frame_height), int(face_bbox.width * frame_width), int(face_bbox.height * frame_height)
    face_img = frame[y:y+h, x:x+w]
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img / 255.0
    face_img = np.reshape(face_img, (1, 48, 48, 1))

    predictions = model.predict(face_img)
    emotion = emotion_labels[np.argmax(predictions)]

    return emotion == 'Angry'

def detect_unusual_activity(faces, hands_landmarks, pose_landmarks, frame):
    for face in faces:
        face_bbox = face.location_data.relative_bounding_box
        face_center = (
            (face_bbox.xmin + face_bbox.width / 2) * frame_width,
            (face_bbox.ymin + face_bbox.height / 2) * frame_height
        )

        # Check hands near face
        for hand_landmarks in hands_landmarks:
            for landmark in hand_landmarks.landmark:
                hand_point = (landmark.x * frame_width, landmark.y * frame_height)
                distance = calculate_distance(face_center, hand_point)
                if distance < threshold_distance:
                    return "Unusual Activity"

        # Check legs near face (e.g., kicking)
        if pose_landmarks:
            left_foot_index = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            right_foot_index = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            left_foot_point = (left_foot_index.x * frame_width, left_foot_index.y * frame_height)
            right_foot_point = (right_foot_index.x * frame_width, right_foot_index.y * frame_height)
            left_distance = calculate_distance(face_center, left_foot_point)
            right_distance = calculate_distance(face_center, right_foot_point)
            if left_distance < threshold_distance or right_distance < threshold_distance:
                return "Unusual Activity"

            # Check hands near neck (approximate neck using shoulders and upper chest landmarks)
            neck_landmarks = [
                pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
                pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            ]
            neck_center = np.mean([(lm.x * frame_width, lm.y * frame_height) for lm in neck_landmarks], axis=0)

            for hand_landmarks in hands_landmarks:
                for landmark in hand_landmarks.landmark:
                    hand_point = (landmark.x * frame_width, landmark.y * frame_height)
                    distance = calculate_distance(neck_center, hand_point)
                    if distance < threshold_distance:
                        return "Unusual Activity"

        # Check for angry face
        if is_angry_face(face_bbox, frame):
            return "Suspicious Activity"

    return "Normal Activity"

def create_video_from_images(image_dir, output_file, extension='jpg'):
    images = [img for img in os.listdir(image_dir) if img.endswith(extension)]
    if not images:
        print("No images found in the directory.")
        return

    images.sort()  # Ensure the images are in chronological order
    first_image_path = os.path.join(image_dir, images[0])
    frame = cv2.imread(first_image_path)
    height, width, channels = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 5.0, (width, height))

    for image in images:
        image_path = os.path.join(image_dir, image)
        frame = cv2.imread(image_path)
        out.write(frame)

    out.release()

try:
    # Initialize video writer for unusual activities
    unusual_activity_video_path = args['output']
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape

        face_results = face_detection.process(frame_rgb)
        hand_results = hands.process(frame_rgb)
        pose_results = pose.process(frame_rgb)

        faces = face_results.detections if face_results.detections else []
        hands_landmarks = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else []
        pose_landmarks = pose_results.pose_landmarks

        activity_label = detect_unusual_activity(faces, hands_landmarks, pose_landmarks, frame)

        exact_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')

        if activity_label == "Unusual Activity":
            img_filename = f"unusual_activities (danger)/unusual_activity_{exact_time}.jpg"
            cv2.imwrite(img_filename, frame)

            # Write frame to video
            if out is None:
                out = cv2.VideoWriter(unusual_activity_video_path, fourcc, 5.0, (frame_width, frame_height))
            out.write(frame)
        elif activity_label == "Suspicious Activity":
            img_filename = f"suspicious_activities/suspicious_activity_{exact_time}.jpg"
            cv2.imwrite(img_filename, frame)
        else:
            img_filename = f"normal_activities/normal_activity_{exact_time}.jpg"
            cv2.imwrite(img_filename, frame)

        for face in faces:
            mp.solutions.drawing_utils.draw_detection(frame, face)

        for hand_landmarks in hands_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Put the label on the frame
        cv2.putText(frame, activity_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Video Surveillance - Unusual Activity Detection", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

finally:
    video.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
