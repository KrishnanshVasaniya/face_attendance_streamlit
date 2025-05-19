# 🧑‍🏫 Face Recognition Attendance System

A Streamlit-based app to manage attendance using face recognition via webcam or image uploads.

## 🚀 Features

- 🔍 Real-time webcam attendance (5 seconds)
- 📤 Upload image (group/selfie) and mark attendance
- ➕ Register new faces via camera or from unknown matches
- ✅ Confidence score displayed for each match
- 📄 Attendance CSV auto-generated

## 📦 Tech Stack

- Streamlit
- OpenCV
- face_recognition (dlib)
- Pandas, Pillow
- Python 3.9+

## 📸 Screenshots

![Webcam Detection](demo1.png)
![Upload & Register](demo2.png)

## 🧠 How to Use

```bash
# 1. Clone repo
git clone https://github.com/<your-username>/face_attendance_streamlit.git
cd face_attendance_streamlit

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```
