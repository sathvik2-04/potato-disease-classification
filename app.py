import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(
    page_title="Potato Disease Detection",
    layout="centered"
)

# Load model
model = load_model("potato.h5", compile=False)

# Class names
class_names = ['Early Blight', 'Late Blight', 'Healthy']

# Title
st.title("Potato Disease Detection")
st.markdown(
    "Upload a potato leaf image to detect whether it has "
    "Early Blight, Late Blight, or is Healthy."
)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png', 'bmp']
)

if uploaded_file is not None:

    # Open image and convert to RGB
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize image
    img_resized = image.resize((224, 224))

    # Convert image to numpy array
    img_array = np.array(img_resized)

    # Normalize image
    img_array = img_array.astype("float32") / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction button
    if st.button("Classify Image"):

        # Predict
        predictions = model.predict(img_array)

        # Get predicted class
        predicted_class = np.argmax(predictions[0])

        # Confidence score
        confidence = np.max(predictions[0]) * 100

        # Show result
        st.success(
            f"Prediction: {class_names[predicted_class]}"
        )

        

else:
    st.info("Please upload a potato leaf image to begin.")

# Footer
st.markdown("---")
st.caption("Potato Disease Detection System")