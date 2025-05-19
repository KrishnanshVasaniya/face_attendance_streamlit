import streamlit as st
import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from deepface import DeepFace
from PIL import Image

# ------------------ Setup ------------------
st.set_page_config(page_title="Face Attendance System", layout="centered")
st.title("🧑‍🏫 Face Attendance System DeepFace")

known_faces_dir = "known_faces"
os.makedirs(known_faces_dir, exist_ok=True)
attendance_csv = "attendance.csv"

# ------------------ Helpers ------------------
def save_face_image(name, image_array):
    path = os.path.join(known_faces_dir, f"{name}.jpg")
    cv2.imwrite(path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    return path

def load_known_faces():
    return [os.path.join(known_faces_dir, f) for f in os.listdir(known_faces_dir) if f.endswith(('.jpg','.png','.jpeg'))]

# ------------------ Register Face ------------------
st.sidebar.header("➕ Register New Face")
name_input = st.sidebar.text_input("Enter name")
image_file = st.sidebar.file_uploader("Upload a clear face photo", type=['jpg', 'png', 'jpeg'])

if st.sidebar.button("Register Face"):
    if name_input and image_file:
        img = Image.open(image_file).convert("RGB")
        img_array = np.array(img)
        save_face_image(name_input.strip(), img_array)
        st.sidebar.success(f"✅ Registered {name_input.strip()}")
    else:
        st.sidebar.error("Please provide name and image.")

# ------------------ Upload & Detect ------------------
st.subheader("📤 Upload Image for Attendance")
img_upload = st.file_uploader("Upload an image with face(s)", type=['jpg','jpeg','png'])

if img_upload:
    uploaded_img = Image.open(img_upload).convert("RGB")
    uploaded_arr = np.array(uploaded_img)
    st.image(uploaded_img, caption="Uploaded Image", use_container_width=True)

    known_faces = load_known_faces()
    if not known_faces:
        st.warning("No known faces registered. Please register first.")
    else:
        attendance_df = pd.read_csv(attendance_csv) if os.path.exists(attendance_csv) else pd.DataFrame(columns=["Name", "Time"])
        today = datetime.now().strftime("%Y-%m-%d")
        marked_names = []

        for face_path in known_faces:
            name = os.path.splitext(os.path.basename(face_path))[0]
            try:
                result = DeepFace.verify(uploaded_arr, face_path, enforce_detection=False)
                if result["verified"] and name not in marked_names:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if not ((attendance_df["Name"] == name) & (attendance_df["Time"].str.contains(today))).any():
                        attendance_df = pd.concat([attendance_df, pd.DataFrame([[name, now]], columns=["Name", "Time"])], ignore_index=True)
                        marked_names.append(name)
                        st.success(f"✅ Attendance marked: {name}")
            except:
                pass

        attendance_df.to_csv(attendance_csv, index=False)

# ------------------ Attendance Log ------------------
st.subheader("📄 Attendance Log")
if os.path.exists(attendance_csv):
    df = pd.read_csv(attendance_csv)
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "attendance.csv", "text/csv")
else:
    st.info("No attendance records yet.")
