# 🧑‍🏫 Face Attendance System (DeepFace + Streamlit)

A simple yet powerful web app that uses **facial recognition** to mark attendance automatically using the [DeepFace](https://github.com/serengil/deepface) library. Built with Python and deployed using **Streamlit Cloud**.

Image.png

---

## 🔥 Features

- 🔐 Register new faces (with name & image)
- 🧠 Face verification using DeepFace
- 📸 Upload an image to detect and mark attendance
- ✅ Auto-prevents duplicate entries for the same day
- 📄 Attendance log with download as CSV
- 🌙 Dark UI theme (customizable)

---

## 🚀 Live Demo

👉 **[Click to try the live app](https://faceattendanceapp-2hzmhrqzsclzdaqjbga78n.streamlit.app/#face-attendance-system-deepface)**  


---

## 📦 Installation (Local)

```bash
# Clone the repo
git clone https://github.com/krishnanshvasaniya/face_attendance_streamlit.git
cd face_attendance_streamlit

# Create a virtual environment (Python 3.10 recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
