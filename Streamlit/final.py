import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import streamlit as st
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix

# Define hierarchical categories
categories = {
    "IAB1 Arts And Entertainment": ["IAB1_1_Books", "IAB1_5_Movies", "IAB1_6_Music"],
    "IAB2 Automotive": ["IAB2_11_Hatchback", "IAB2_14_MiniVan", "IAB2_15_Mororcycles", "IAB2_21_Trucks  Accessories"],
    "IAB3 Business": ["IAB3_6_Forestry"],
    "IAB17 Sports": [
        "IAB17_5_Boxing", "IAB17_6_CanoeING", "IAB17_7_Cheerleading", "IAB17_9_Cricket", "IAB17_11_Fly Fishing",
        "IAB17_12_Football", "IAB17_15_Golf", "IAB17_16_Horse Racing", "IAB17_29_Rugby", "IAB17_31_SailING",
        "IAB17_36_Snowboarding", "IAB17_37_Surfing", "IAB17_38_Swimming", "IAB17_39_Table Tennis", "IAB17_40_Tennis",
        "IAB17_41_Volleyball"
    ]
}

# App title
st.title("Bi-Level Image Classification")

# Sidebar options
st.sidebar.title("Options")
model_path = st.sidebar.text_input("Enter the path of the saved .h5 model:", "D:\\Sem\\Sem 5\\SGP-II\\IAB\\Streamlit\\ResNet101V2_E100.h5")
test_dir = st.sidebar.text_input("Enter the path to the test directory:", "D:\\Sem\\Sem 5\\SGP-II\\IAB\\Dataset\\Output\\test")

# Load the model
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

try:
    model = load_model(model_path)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# Load test data
def get_test_data_generator(test_dir):
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
    return test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )

if os.path.exists(test_dir):
    test_generator = get_test_data_generator(test_dir)
    class_labels = list(test_generator.class_indices.keys())
else:
    st.error("Invalid test directory path.")

# Evaluate model
if st.sidebar.button("Evaluate Model"):
    try:
        test_loss, test_accuracy = model.evaluate(test_generator)
        st.write(f"Test Accuracy: {test_accuracy:.2f}")

        # Generate predictions
        predictions = model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes

        # Confusion matrix
        conf_matrix = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        st.pyplot(plt.gcf())

    except Exception as e:
        st.error(f"Error during evaluation: {e}")

# Single image prediction
st.sidebar.subheader("Image Classification")
uploaded_file = st.sidebar.file_uploader("Upload an image to classify:", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        # Read and preprocess the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (224, 224))
        image_processed = preprocess_input(image_resized)
        image_processed = np.expand_dims(image_processed, axis=0)

        # Make prediction
        prediction = model.predict(image_processed)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]

        # Find main category
        parent_category = None
        for parent, subcategories in categories.items():
            if predicted_class in subcategories:
                parent_category = parent
                break

        # Display results
        st.image(image_rgb, caption="Uploaded Image", use_column_width=True)
        st.write(f"**Sub Category:** {predicted_class}")
        st.write(f"**Main Category:** {parent_category}")
    except Exception as e:
        st.error(f"Error during classification: {e}")
