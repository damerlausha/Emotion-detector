import streamlit as st 
import numpy as np 
import cv2
from PIL import Image
from deepface import DeepFace
from deepface.modules.exceptions import FaceNotDetected

st.set_page_config(page_title="Emotion Detection",page_icon="😊")
if not hasattr(cv2, "CascadeClassifier"):
    st.error("This Streamlit process has an incompatible OpenCV module loaded. Stop the running app and start it again.")
    st.stop()

st.title("😊 Live Emotion Detection")
pic=st.camera_input("📷 Take a Picture")
if pic:
    img=Image.open(pic).convert("RGB")
    st.image(img,caption="Captured Image",use_container_width=True)
    if st.button("🔍 Detect Emotion"):
        try:
            result=DeepFace.analyze(
                cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR),
                actions=["emotion"],
                detector_backend="opencv",
                enforce_detection=True
            )[0]
        except FaceNotDetected:
            st.error("❌ No face detected.")
            st.info("Take a clear picture of your face and try again.") 
            st.stop()

        emotion,scores=result["dominant_emotion"],result["emotion"]
        emoji={
            "happy": "😊",
            "sad": "😢",
            "angry": "😡",
            "surprised": "😲",
            "neutral": "😐",
            "fear": "😨",
            "disgust": "🤢"
        }.get(emotion,"🙂")
        st.success(f"{emoji} Emotion:{emotion.upper()}")
        st.subheader("📊 Emotion Scores")
        for name, score in scores.items():
            st.write(f"{name.title()}-{score:.2f}%")
            st.progress(int(score))