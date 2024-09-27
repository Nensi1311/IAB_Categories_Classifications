# Step-1 install required libraries

!pip install tensorflow
!pip install numpy
!pip install scikit-learn
!pip install matplotlib


# Step-2 Load the Dataset and Preprocess Images

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image dimensions
img_width, img_height = 224, 224  # Resize images to match input size of common models
batch_size = 32

# Set the paths to your train and test datasets
train_dataset_dir = 'E:/College/Project/IAB/Output/train'  # Adjust path
test_dataset_dir = 'E:/College/Project/IAB/Output/test'    # Adjust path

# Prepare ImageDataGenerator for train and test data
datagen = ImageDataGenerator(rescale=1./255)

# Train generator (use the train split)
train_generator = datagen.flow_from_directory(
    train_dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
)

# Validation generator (use the test split for validation)
validation_generator = datagen.flow_from_directory(
    test_dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
)

# Print class indices to map subcategories
class_indices = train_generator.class_indices
print(class_indices)  # Check the mappings of subcategories to numeric labels

