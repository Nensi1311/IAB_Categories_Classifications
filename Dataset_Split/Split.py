import os
import shutil
from sklearn.model_selection import train_test_split

# Paths to your dataset and output folders
dataset_dir = 'E:/College/Project/IAB/Final_IAB'  # Adjust the path
output_dir = 'E:/College/Project/IAB/Output'      # Adjust the path

# Train and test directories
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')

# Function to create directories if they don't exist
def create_dirs(base_dir, category, subfolder):
    os.makedirs(os.path.join(base_dir, category, subfolder), exist_ok=True)

# Function to split data into train and test sets
def split_data(category_path, category, subfolder, train_dir, test_dir):
    subfolder_path = os.path.join(category_path, subfolder)
    image_filenames = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]
    
    # Split train and test
    train_files, test_files = train_test_split(image_filenames, test_size=0.2, random_state=42)
    
    # Copy files to train and test directories
    for filename in train_files:
        src_file = os.path.join(subfolder_path, filename)
        dest_file = os.path.join(train_dir, category, subfolder, filename)
        shutil.copy(src_file, dest_file)
    
    for filename in test_files:
        src_file = os.path.join(subfolder_path, filename)
        dest_file = os.path.join(test_dir, category, subfolder, filename)
        shutil.copy(src_file, dest_file)

# Get categories (main folders like IAB1, IAB2, etc.)
categories = os.listdir(dataset_dir)

# Traverse through each category and its subfolders
for category in categories:
    category_path = os.path.join(dataset_dir, category)
    
    # Ensure we are dealing with directories only
    if os.path.isdir(category_path):
        subfolders = os.listdir(category_path)
        
        # Create directories in train and test folders
        for subfolder in subfolders:
            create_dirs(train_dir, category, subfolder)
            create_dirs(test_dir, category, subfolder)
            
            # Split and copy data from subfolders
            split_data(category_path, category, subfolder, train_dir, test_dir)

print("Data split into train and test completed, preserving subfolder structure.")

