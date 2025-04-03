import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("waste_classifier_colab.h5", compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = load_model()

# Class mapping and recommendations
CLASS_MAPPING = {
    0: 'cardboard',
    1: 'glass',
    2: 'metal',
    3: 'paper',
    4: 'plastic',
    5: 'trash'
}

RECOMMENDATIONS = {
    'cardboard': 'Recycle in cardboard bin',
    'glass': 'Recycle in glass container',
    'metal': 'Recycle in metal bin',
    'paper': 'Recycle in paper bin',
    'plastic': 'Check local guidelines',
    'trash': 'General waste disposal'
}

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image).convert('RGB').resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# Streamlit UI
st.title("Waste Classification App")
st.write("Upload an image to classify the type of waste and get disposal recommendations.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    try:
        img_tensor = preprocess_image(uploaded_file)
        predictions = model.predict(img_tensor, verbose=0)
        predicted_idx = np.argmax(predictions, axis=1)[0]
        predicted_class = CLASS_MAPPING[predicted_idx]
        confidence = float(predictions[0][predicted_idx])
        recommendation = RECOMMENDATIONS[predicted_class]

        st.write(f"**Prediction:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2%}")
        st.write(f"**Recommendation:** {recommendation}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
