# 👤 Face Authentication with Liveness Detection

Real-time face authentication system using:
- **Blink + Head Turn Liveness Detection** (Mediapipe)
- **Face Embedding** (DeepFace + Facenet)
- **Similarity Check** (cosine similarity)

## 🧠 Features
- Live camera input with OpenCV
- Blinking and yaw-based liveness check
- Facial feature extraction and embedding
- Local embedding database (saved as `.pkl`)
- Cosine similarity-based authentication

## 🗂️ Project Structure
├── local_face_authf.py # Main app
├── database/ # Saved user images
├── *.dat / *.caffemodel # Pre-trained models (ignored)
├── .gitignore
├── requirements.txt

## 🧪 Usage
```bash
python local_face_authf.py
Options:
- **1**: Add user to database
- **2**: Authenticate user
- **3**: Exit