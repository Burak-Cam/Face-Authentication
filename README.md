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

| File                                            | Description                     | Link                                                                                                                                                       |
| ----------------------------------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `shape_predictor_68_face_landmarks.dat`         | Facial landmark predictor       | [Download](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)                                                                                |
| `dlib_face_recognition_resnet_model_v1.dat`     | Dlib face embedding model       | [Download](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)                                                                            |
| `res10_300x300_ssd_iter_140000_fp16.caffemodel` | OpenCV face detector            | [Download](https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel) |
| `deploy.prototxt.txt`                           | Config for OpenCV face detector | [Download](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)                                                         |

📌 **Note:** `.bz2` files need to be extracted before use.

---

## 📜 License

This project is shared for educational purposes.
If you use or modify this code, credit is appreciated.

---

## 🖼️ Demo

You can in

* **Burak Çam**
* [github.com/Burak-Cam](https://github.com/Burak-Cam)
