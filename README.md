# 👤 Face Authentication with Liveness Detection

A real-time face authentication system built with Python.  
Includes **blink & head turn-based liveness detection**, facial feature extraction, and cosine similarity-based verification.

> 🔒 Built using: OpenCV, Dlib, Mediapipe, DeepFace, NumPy  
> 📦 Models excluded from repo (see below)

---

## 📌 Features

- Blink + head yaw detection for liveness
- Facial embeddings with DeepFace (Facenet)
- User enrollment via webcam
- Cosine similarity-based authentication
- Local database of feature vectors

---

## 📁 Project Structure

face_auth/

├── local_face_authf.py # Main app

├── requirements.txt # Dependencies

├── .gitignore

├── README.md

├── database/ # Contains captured user face images

└── models/ # Put model files here (see below)


---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the main script:

```bash
python local_face_authf.py
```

3. Choose:

   * `1` → Add a new user to the database (capture face frames)
   * `2` → Authenticate an existing user
   * `3` → Exit

---

## 🧪 Liveness Detection

Liveness check requires:

* **Blink detection** using Eye Aspect Ratio (EAR)
* **Head turn detection** using yaw angle from Mediapipe face mesh

User is verified only after passing both checks.

---

## ✅ Example Output

```text
Please turn your head LEFT and blink...
✅ Blink detected!
✅ Head turned left!
Liveness check passed ✅
Authenticating user...
Similarity: 0.91 ✅ Authenticated
```

---

## 📦 Model Downloads

Download the following pre-trained models and place them in the project directory (or a `models/` folder):

| File                                            | Description                     | Link                                                                                                                             |
| ----------------------------------------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `shape_predictor_68_face_landmarks.dat`         | Facial landmark predictor       | [Download](https://github.com/ageitgey/face_recognition_models/blob/e67de717267507d1e9246de95692eb8be736ab61/face_recognition_models/models/shape_predictor_68_face_landmarks.dat)                                                      |
| `dlib_face_recognition_resnet_model_v1.dat`     | Dlib face embedding model       | [Download](https://github.com/ageitgey/face_recognition_models/blob/e67de717267507d1e9246de95692eb8be736ab61/face_recognition_models/models/dlib_face_recognition_resnet_model_v1.dat)                                                  |
| `res10_300x300_ssd_iter_140000_fp16.caffemodel` | OpenCV face detector            | [Download](https://github.com/sr6033/face-detection-with-OpenCV-and-DNN/blob/master/res10_300x300_ssd_iter_140000.caffemodel) |
| `deploy.prototxt.txt`                           | Config for OpenCV face detector | [Download](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)                               |


---

## 📜 License

This project is shared for educational purposes.
If you use or modify this code, credit is appreciated.

---

## 🙋 Author

* **Burak Çam**
* [github.com/Burak-Cam](https://github.com/Burak-Cam)
