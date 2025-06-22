import cv2
import dlib
import numpy as np
import os
import pickle
import mediapipe as mp
import time
import random
from collections import deque
from deepface import DeepFace

# Model and file paths
shape_predictor = dlib.shape_predictor(r"C:\Users\burak\face_auth\shape_predictor_68_face_landmarks.dat")
net = cv2.dnn.readNetFromCaffe(
    r"C:\Users\burak\face_auth\deploy.prototxt.txt",
    r"C:\Users\burak\face_auth\res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# Mediapipe settings
mp_face_mesh = mp.solutions.face_mesh
EAR_HISTORY_LEN = 20
LEFT_EYE_IDS = [33, 160, 158, 133, 153, 144]  # Left eye landmarks
RIGHT_EYE_IDS = [362, 385, 387, 263, 373, 380]  # Right eye landmarks

def eye_aspect_ratio(eye):
    """Calculate the eye aspect ratio to detect blinks."""
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

def get_head_pose_yaw(landmarks, image_shape):
    """Returns yaw angle (left/right head rotation) in degrees, reversed for mirror effect."""
    h, w = image_shape[:2]

    # 2D image points
    image_points = np.array([
        [landmarks[1].x * w, landmarks[1].y * h],     # Nose tip
        [landmarks[33].x * w, landmarks[33].y * h],   # Left eye outer corner
        [landmarks[263].x * w, landmarks[263].y * h], # Right eye outer corner
        [landmarks[61].x * w, landmarks[61].y * h],   # Mouth left
        [landmarks[291].x * w, landmarks[291].y * h], # Mouth right
        [landmarks[199].x * w, landmarks[199].y * h], # Chin
    ], dtype="double")

    # 3D model points (approximate)
    model_points = np.array([
        [0.0, 0.0, 0.0],        # Nose tip
        [-30.0, -30.0, -30.0],  # Left eye corner
        [30.0, -30.0, -30.0],   # Right eye corner
        [-25.0, 30.0, -30.0],   # Mouth left
        [25.0, 30.0, -30.0],    # Mouth right
        [0.0, 70.0, -50.0],     # Chin
    ])

    # Camera internals
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    # SolvePnP
    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat([rotation_mat, np.zeros((3, 1))])
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    yaw = -euler_angles[1][0]  # Reverse yaw to compensate for mirror effect
    return yaw

def wait_for_blink():
    """Perform liveness check: Detect blink and head turn using yaw angle."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera could not be opened.")
        return None

    head_direction = random.choice(["right", "left"])
    print(f"Please turn your head {head_direction.upper()} about 30-45 degrees and blink within 15 seconds!")

    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Preview", 640, 480)

    blink_confirmed = 0
    blink_detected = False
    head_detected = False
    blink_reported = False
    head_reported = False

    start_time = time.time()
    ear_hist = deque(maxlen=20)
    liveness_frame = None

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    ) as face_mesh:
        while time.time() - start_time < 15 and not (blink_detected and head_detected):
            ret, frame = cap.read()
            if not ret:
                print("⚠️ No frame captured, check camera connection.")
                continue

            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(img_rgb)
            remaining = int(15 - (time.time() - start_time))
            ear_avg = 1.0  # Default value if no face detected

            if not res.multi_face_landmarks:
                print("⚠️ No face detected, please position your face in the frame.")
                cv2.imshow("Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            lm = res.multi_face_landmarks[0]
            coords = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]

            # EAR calculation
            left_eye = [coords[i] for i in LEFT_EYE_IDS]
            right_eye = [coords[i] for i in RIGHT_EYE_IDS]
            ear_avg = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            ear_hist.append(ear_avg)

            if ear_avg < 0.20:
                blink_confirmed += 1
                if blink_confirmed >= 2 and not blink_reported:
                    print(f"✅ Blink detected! EAR: {ear_avg:.2f}")
                    blink_detected = True
                    blink_reported = True
            else:
                blink_confirmed = 0

            # Head pose (yaw)
            yaw_angle = get_head_pose_yaw(lm.landmark, frame.shape)
            if yaw_angle is not None:
                cv2.putText(frame, f"Yaw: {yaw_angle:.2f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                threshold_yaw = 15.0
                if head_direction == "right" and yaw_angle > threshold_yaw and not head_reported:
                    print(f"✅ Head turned right! Yaw: {yaw_angle:.2f}")
                    head_detected = True
                    head_reported = True
                elif head_direction == "left" and yaw_angle < -threshold_yaw and not head_reported:
                    print(f"✅ Head turned left! Yaw: {yaw_angle:.2f}")
                    head_detected = True
                    head_reported = True

            # UI
            cv2.putText(frame, f"EAR: {ear_avg:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Time left: {remaining}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Turn head: {head_direction}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(frame, f"Threshold: {threshold_yaw}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if blink_detected and head_detected and liveness_frame is None:
                liveness_frame = frame.copy()
                print("Liveness check passed, capturing initial frame.")

    if not (blink_detected and head_detected):
        print("❌ Liveness check failed: Timed out or conditions not met.")
        cap.release()
        if cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow("Preview")
        cv2.waitKey(1)
        return None

    # Prompt user to return to frontal position
    print("Please return your face to a frontal position and hold for 2 seconds...")
    start_time = time.time()
    while time.time() - start_time < 2:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow("Preview")  # Destroy window only if it exists
    cv2.waitKey(1)

    print("✅ Liveness check passed, using frontal frame.")
    return frame  # Returns the frontal frame

def extract_features(image):
    """Extract facial features using DeepFace with temporary file workaround."""
    temp_path = "temp_frame.jpg"
    try:
        print("Extracting features with DeepFace...")
        cv2.imwrite(temp_path, image)
        embedding = DeepFace.represent(img_path=temp_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        print(f"Feature extraction successful, length: {len(embedding)}")
        return np.array(embedding)
    except Exception as e:
        print(f"❌ Feature extraction failed: {str(e)}")
        return None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)  # Ensure cleanup even on error

def build_database(user_name, database_path="./database"):
    """Save user facial features to the database."""
    user_dir = os.path.join(database_path, user_name)
    features = []
    
    if os.path.exists(user_dir):
        for img_name in os.listdir(user_dir):
            img_path = os.path.join(user_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            feature = extract_features(img)
            if feature is not None:
                features.append(feature)
    
    if len(features) > 0:
        with open(f"{database_path}/{user_name}_features.pkl", "wb") as f:
            pickle.dump(features, f)
        print(f"✅ Saved {len(features)} features for {user_name}.")
    else:
        print(f"❌ No features extracted for {user_name}.")

def authenticate_identity(user_name, database_path="./database"):
    """Authenticate user using DeepFace."""
    print(f"Authenticating {user_name}...")
    features_path = f"{database_path}/{user_name}_features.pkl"
    if not os.path.exists(features_path):
        print(f"❌ No database found for {user_name}")
        return False

    with open(features_path, "rb") as f:
        db_features = pickle.load(f)
    print(f"Loaded {len(db_features)} features from database")

    liveness_frame = wait_for_blink()
    if liveness_frame is None:
        print("❌ Liveness check failed")
        return False

    feature = extract_features(liveness_frame)
    if feature is None:
        print("❌ Feature extraction failed")
        return False

    # Calculate similarity (cosine similarity)
    similarities = [np.dot(feature, db_feature) / (np.linalg.norm(feature) * np.linalg.norm(db_feature))
                    for db_feature in db_features if np.linalg.norm(db_feature) > 0]
    print(f"Calculated {len(similarities)} similarities")
    max_similarity = max(similarities) if similarities else 0.0
    threshold = 0.85
    print(f"Debug: Max Similarity = {max_similarity:.4f}, Threshold = {threshold:.4f}")

    if max_similarity > threshold:
        print(f"✅ Authentication successful: {user_name} (Similarity: {max_similarity:.4f})")
        return True
    else:
        print(f"❌ Authentication failed (Similarity: {max_similarity:.4f})")
        return False

def main():
    """Main menu for the face authentication system."""
    database_path = "./database"
    while True:
        print("\n1. Add frames to database")
        print("2. Start authentication")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")
        if choice == "1":
            user_name = input("Enter user name: ")
            user_dir = os.path.join(database_path, user_name)
            if os.path.exists(user_dir) and len(os.listdir(user_dir)) >= 100:
                print(f"Enough frames exist for {user_name}, updating database...")
                build_database(user_name, database_path)
            else:
                print("Starting frame capture with camera...")
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("❌ Camera could not be opened.")
                    continue
                os.makedirs(user_dir, exist_ok=True)
                count = 0
                while count < 100:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.imwrite(os.path.join(user_dir, f"{user_name}_frame_{count}.jpg"), frame)
                    cv2.imshow("Frame Capture", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    count += 1
                cap.release()
                if cv2.getWindowProperty("Frame Capture", cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow("Frame Capture")
                print(f"{count} frames captured.")
                build_database(user_name, database_path)
        elif choice == "2":
            user_name = input("Enter user name: ")
            authenticate_identity(user_name, database_path)
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()