import streamlit as st
import cv2
import numpy as np

st.title("Face Detection App")
# Instructions
st.markdown("""
### How to Use This App
1. Choose an image source: upload an image OR use the webcam.  
2. Adjust detection settings (scaleFactor and minNeighbors).  
3. Choose the rectangle color for detected faces.  
4. View the processed image.  
5. Download the output if needed.
""")

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Detection settings
scale = st.slider("Select scaleFactor", 1.01, 2.0, 1.1)
neighbors = st.slider("Select minNeighbors", 1, 20, 5)

# Rectangle color picker
color_hex = st.color_picker("Pick rectangle color", "#00FF00")
color = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))  # HEX → BGR

# Choose mode: Upload or Webcam

mode = st.radio("Choose input source:", ["Upload Image", "Use Webcam"])

img = None

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

elif mode == "Use Webcam":
    webcam_image = st.camera_input("Take a photo")
    if webcam_image:
        img_bytes = np.frombuffer(webcam_image.getvalue(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)


# Process Image

if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale,
        minNeighbors=neighbors
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Processed Image")

    # Save output
    output_path = "detected_faces.jpg"
    cv2.imwrite(output_path, img)

    with open(output_path, "rb") as f:
        st.download_button(
            label="Download Image",
            data=f,
            file_name="detected_faces.jpg",
            mime="image/jpeg"
        )

