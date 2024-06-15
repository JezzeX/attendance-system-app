import streamlit as st
from Home import face_rec
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

st.subheader("Registration Form")

registration_form = face_rec.RegistrationForm()


def video_callback_func(frame):
    img = frame.to_ndarray(format="bgr24")
    reg_img, embedding = registration_form.get_embedding(img)
    if embedding is not None:
        with open("face_embedding.txt", mode="ab") as f:
            np.savetxt(f, embedding)
    return av.VideoFrame.from_ndarray(reg_img, format="bgr24")


with st.container():
    name = st.text_input(label="Name", placeholder="Enter First name and Last name")
    st.write("Click on Start button to collect your face samples")
    with st.expander("Instructions"):
        st.caption("1. Give different expressions to capture your face details.")
        st.caption("2. Click on stop after getting 200 samples.")
    webrtc_streamer(
        key="registration",
        video_frame_callback=video_callback_func,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

if st.button("Submit"):
    return_val = registration_form.save_data_in_redis_db(name)
    if return_val == True:
        st.success(f"{name} registered successfully")
    elif return_val == "name_false":
        st.error("Please enter the name: Name cannot be empty or spaces")
    elif return_val == "file_false":
        st.error(
            "face_embedding.txt is not found. Please refresh the page and execute again."
        )

unenroll_name = st.text_input(
    label="Name to Unenroll", placeholder="Enter the name to unenroll"
)
if st.button("Unenroll"):
    return_val = registration_form.delete_data_in_redis_db(unenroll_name)
    if return_val:
        st.success(f"{unenroll_name} unenrolled successfully")
    else:
        st.error("Error unenrolling. Please check the name and try again.")
