import streamlit as st
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd
import time
from PIL import Image

# Page setup
st.set_page_config(page_title="Face Attendance System", layout="centered")
st.title("🧑‍🏫 Face Recognition Attendance System")

# Constants
known_faces_dir = "known_faces"
attendance_csv = "attendance.csv"
os.makedirs(known_faces_dir, exist_ok=True)

# Load known faces
def load_known_faces():
    encodings = []
    names = []
    for file in os.listdir(known_faces_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(known_faces_dir, file)
            img = face_recognition.load_image_file(img_path)
            face_encs = face_recognition.face_encodings(img)
            if face_encs:
                encodings.append(face_encs[0])
                names.append(os.path.splitext(file)[0])
    return encodings, names

known_encodings, known_names = load_known_faces()

# -----------------------------------
# 📌 SIDEBAR: Register New Face
# -----------------------------------
st.sidebar.header("➕ Register New Face")
new_name = st.sidebar.text_input("Enter Name")
start_cam_reg = st.sidebar.button("Start Camera to Register")

if "reg_frame" not in st.session_state:
    st.session_state.reg_frame = None

if start_cam_reg:
    if not new_name.strip():
        st.sidebar.error("Please enter a name first!")
    else:
        cap = cv2.VideoCapture(0)
        st.sidebar.write("📸 Capturing image for registration...")
        time.sleep(1)
        _, frame = cap.read()
        st.session_state.reg_frame = frame
        cap.release()

        if frame is not None:
            st.sidebar.image(frame, channels="BGR", caption="Captured Image")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if face_locations:
                save_path = os.path.join(known_faces_dir, f"{new_name}.jpg")
                cv2.imwrite(save_path, frame)
                st.sidebar.success(f"✅ Registered: {new_name}")
                known_encodings, known_names = load_known_faces()
            else:
                st.sidebar.error("❌ No face detected. Try again with a clearer image.")

# -----------------------------------
# 📤 Upload Image for Testing (Offline Mode)
# -----------------------------------
st.subheader("📤 Upload Image for Offline Attendance")
uploaded_image = st.file_uploader("Upload a group or selfie image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)

    if image.width > 1000:
        image = image.resize((800, int(800 * image.height / image.width)))

    image = image.convert("RGB")
    rgb_img = np.array(image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    attendance_df = pd.read_csv(attendance_csv) if os.path.exists(attendance_csv) else pd.DataFrame(columns=["Name", "Time"])
    today = datetime.now().strftime("%Y-%m-%d")
    captured_names = []

    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
    st.write(f"🧠 Detected {len(face_encodings)} face(s) in uploaded image.")

    # Draw bounding boxes on copy of image
    output_img = rgb_img.copy()

    for idx, (top, right, bottom, left) in enumerate(face_locations):
        face_encoding = face_encodings[idx]
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            if best_distance < 0.6:
                name = known_names[best_match_index]
                st.success(f"✅ Match found: {name} (Confidence: {(1 - best_distance):.2f})")

                if name not in captured_names:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if not ((attendance_df["Name"] == name) & (attendance_df["Time"].str.contains(today))).any():
                        new_row = pd.DataFrame([{"Name": name, "Time": now}])
                        attendance_df = pd.concat([attendance_df, new_row], ignore_index=True)
                        captured_names.append(name)
                        st.success(f"📝 Attendance marked: {name}")
            else:
                st.warning(f"❌ Face #{idx+1} not recognized (Low confidence: {(1 - best_distance):.2f})")

        # Draw box
        cv2.rectangle(output_img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(output_img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if name == "Unknown":
            with st.form(key=f"register_form_{idx}"):
                st.warning(f"👤 Unknown face #{idx+1} detected!")
                new_name_unknown = st.text_input(f"Enter name for face #{idx+1}", key=f"input_{idx}")
                submit = st.form_submit_button(f"Register Face #{idx+1}")
                if submit and new_name_unknown.strip():
                    face_img = rgb_img[top:bottom, left:right]
                    save_path = os.path.join(known_faces_dir, f"{new_name_unknown.strip()}.jpg")
                    cv2.imwrite(save_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                    st.success(f"✅ Registered {new_name_unknown} successfully!")
                    known_encodings, known_names = load_known_faces()

    st.image(output_img, caption="📸 Detected Faces", channels="RGB", use_container_width=True)

# -----------------------------------
# 🎥 Webcam Attendance (5 seconds)
# -----------------------------------
st.subheader("📷 Live Webcam Attendance (5 seconds)")

if st.button("Start Attendance Capture"):
    st.info("🎬 Starting webcam...")

    cap = cv2.VideoCapture(0)
    start_time = time.time()
    captured_names = []

    attendance_df = pd.read_csv(attendance_csv) if os.path.exists(attendance_csv) else pd.DataFrame(columns=["Name", "Time"])

    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for idx, (top, right, bottom, left) in enumerate(face_locations):
            face_encoding = face_encodings[idx]
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match = np.argmin(face_distances)
                if face_distances[best_match] < 0.6:
                    name = known_names[best_match]

                    if name not in captured_names:
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        today = now.split(" ")[0]
                        if not ((attendance_df["Name"] == name) & (attendance_df["Time"].str.contains(today))).any():
                            new_row = pd.DataFrame([{"Name": name, "Time": now}])
                            attendance_df = pd.concat([attendance_df, new_row], ignore_index=True)
                            captured_names.append(name)
                            st.success(f"✅ Attendance marked: {name}")

            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        st.image(frame, channels="BGR", caption="📸 Detected Faces")
        time.sleep(0.3)

    cap.release()
    attendance_df.to_csv(attendance_csv, index=False)
    st.success("📝 Attendance saved!")

# -----------------------------------
# 📄 Attendance Log
# -----------------------------------
st.subheader("📄 Attendance Log")

if os.path.exists(attendance_csv):
    df = pd.read_csv(attendance_csv)
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "attendance.csv", "text/csv")
else:
    st.info("No attendance has been marked yet.")
