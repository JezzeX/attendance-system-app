import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time
import numpy as np
import cv2
from PIL import Image

st.subheader("Real-Time Attendance System")

with st.spinner("Retrieving Data from Redis DB ..."):
    redis_face_db = face_rec.retrive_data(name="academy:register")
    st.dataframe(redis_face_db)

st.success("Data successfully retrieved from Redis")

waitTime = 30
setTime = time.time()
realtimepred = face_rec.RealTimePred()


def video_frame_callback(frame):
    global setTime
    img = frame.to_ndarray(format="bgr24")
    pred_img = realtimepred.face_prediction(
        img, redis_face_db, "facial_features", "name", thresh=0.5
    )
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(
    key="realtimePrediction",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)


st.subheader("Upload Photos for Attendance")

uploaded_files = st.file_uploader(
    "Choose a photo...", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        img_array = np.array(img)
        with st.spinner(f"Processing {uploaded_file.name}..."):
            pred_img, attendance_data = realtimepred.process_uploaded_image(
                img_array,
                redis_face_db,
                "facial_features",
                "name",
                thresh=0.5,
            )
            st.image(
                pred_img,
                caption=f"Processed {uploaded_file.name}",
                use_column_width=True,
            )
            for record in attendance_data:
                st.write(f"Recorded: {record['name']} at {record['current_time']}")
            realtimepred.saveLogs_redis()
