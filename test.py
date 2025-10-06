#  ----- Prerequisite ----- #
# !!!!!!!!!!!!!!!!!!!!!!!!! #

# To make the code below work, you need to download the pre-trained weights from:
# run : git clone https://github.com/nmfadil/FER-Pretrained-MiniXception.git



# ----- Imports -----

import gradio as gr # For the web app
import cv2 # For webcam access

import tensorflow as tf # for cnn
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from skimage.feature import local_binary_pattern, hog # For LBP and HOG
from sklearn.neighbors import KNeighborsClassifier # For KNN
from sklearn.svm import LinearSVC # For SVM
from sklearn.preprocessing import StandardScaler # For feature scaling
import kagglehub # For loading datasets
import os # For file paths
from PIL import Image # For opening images 
import numpy as np # For array conversion
from tqdm import tqdm # For progress bars

# ----- Face croping ----- #

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_face(frame):
    """Detect and crop the face while maintaining aspect ratio"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        # Get the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Add padding around the face (20%)
        padding = int(max(w, h) * 0.2)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2*padding)
        h = min(frame.shape[0] - y, h + 2*padding)
        
        # Crop the face
        face = frame[y:y+h, x:x+w]
        
        # Resize while maintaining aspect ratio
        target_size = max(TARGET_SIZE)
        aspect_ratio = w/h
        
        if aspect_ratio > 1:
            new_w = target_size
            new_h = int(target_size / aspect_ratio)
        else:
            new_h = target_size
            new_w = int(target_size * aspect_ratio)
            
        face_resized = cv2.resize(face, (new_w, new_h))
        
        # Create a square image with padding
        square_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        square_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = face_resized
        
        return True, square_img
    return False, frame


# ===================== #
# ===== LBP + kNN ===== #
# ===================== #


# Download latest version
path = kagglehub.dataset_download("msambare/fer2013")

# Variables for LBP
R = 1  # Radius
P = 8 * R  # Number of points
n_bins = P + 2  # For uniform LBP histogram
lbps = []
emotions = []

neigh = KNeighborsClassifier(n_neighbors=50, weights='distance')

# Loop on every training image
training_path = os.path.join(path, "train")
for emotion_directory in tqdm(os.listdir(training_path), desc="LBP + kNN Training"):
    emotion_path = os.path.join(training_path, emotion_directory)
    
    for image_name in os.listdir(emotion_path):
        image_path = os.path.join(emotion_path, image_name)
        if os.path.isfile(image_path):
            try:
                # Open the image
                with Image.open(image_path) as img:
                    # Convert PIL image to numpy array
                    img_array = np.array(img, dtype=np.int16)
                    
                    # Apply LBP with uniform patterns
                    lbp = local_binary_pattern(img_array, P, R, method='uniform')
                    
                    # Calculate the histogram of LBP
                    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
                    
                    # Save the histogram and corresponding emotion
                    lbps.append(hist)
                    emotions.append(emotion_directory)

            except Exception as e:
                print(f"Error processing {image_name}: {str(e)}")

# Fit the KNN with all the histograms
neigh.fit(lbps, emotions)


def process_knn(image):
    # Convert PIL image to numpy array with float64 dtype
    img_array = np.array(image, dtype=np.int16)
    
    # Apply LBP with uniform patterns
    lbp = local_binary_pattern(img_array, P, R, method='uniform')
    
    # Calculate the histogram of LBP
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    # Predict the emotion using the KNN
    prediction = neigh.predict([hist])

    # Convert back to uint8 for saving
    lbp_normalized = ((lbp - lbp.min()) * (255 / (lbp.max() - lbp.min()))).astype(np.uint8)

    # Convert numpy array back to PIL Image
    new_img = Image.fromarray(lbp_normalized)

    return new_img, prediction[0]


# ===================== #
# ===== HOG + SVM ===== #
# ===================== #

# HOG parameters

cell_size = (8, 8)
block_size = (2, 2)
orientations = 9

# Initialize SVM classifier and feature scaler
svm_classifier = LinearSVC(random_state=42)
scaler = StandardScaler()

# Lists to store HOG features and emotions
hog_features = []
hog_emotions = []

# Train HOG + SVM
training_path = os.path.join(path, "train")
for emotion_directory in tqdm(os.listdir(training_path), desc="HOG + SVM Training"):
    emotion_path = os.path.join(training_path, emotion_directory)
    
    for image_name in os.listdir(emotion_path):
        image_path = os.path.join(emotion_path, image_name)
        if os.path.isfile(image_path):
            try:
                # Open and process the image
                with Image.open(image_path) as img:

                    img_array = np.array(img, dtype=np.float64)
                    
                    # Calculate HOG features
                    features = hog(img_array,
                                 orientations=orientations,
                                 pixels_per_cell=cell_size,
                                 cells_per_block=block_size,
                                 block_norm='L2-Hys',
                                 feature_vector=True)
                    
                    # Store features and emotion
                    hog_features.append(features)
                    hog_emotions.append(emotion_directory)
            
            except Exception as e:
                print(f"Error processing {image_name}: {str(e)}")

# Scale features and train SVM
hog_features_scaled = scaler.fit_transform(hog_features)
print("Fitting HOG results into SVM ... (might take 2-3 minutes)")
svm_classifier.fit(hog_features_scaled, hog_emotions)
print("Done")

def process_hog(image):
    # Convert image to numpy array with float64 precision for gradient computation
    img_array = np.array(image, dtype=np.float64)
    
    features = hog(img_array,
                  orientations=orientations,    # Number of orientation bins (9)
                  pixels_per_cell=cell_size,   # Cell size for histograms (8x8)
                  cells_per_block=block_size,  # Block size for normalization (2x2 cells)
                  block_norm='L2-Hys',         # Normalization method
                  feature_vector=True,         # Concatenate into 1D feature vector
                  visualize=True)              # Get visualization
    
    # Separate features and visualization
    hog_features, hog_image = features
    
    # Scale features
    hog_features_scaled = scaler.transform([hog_features])
    
    # Predict emotion
    prediction = svm_classifier.predict(hog_features_scaled)
    
    # Normalize HOG visualization for display
    hog_image = (hog_image * 255).astype(np.uint8)
    
    # Convert to PIL Image
    hog_image = Image.fromarray(hog_image)
    
    return hog_image, prediction[0]


# ============================= #
# ===== mini-Xception cnn ===== #
# ============================= #

# Load the pre-trained emotion detection model (adjust the path to where you have it stored)
model = load_model(os.path.join("FER-Pretrained-MiniXception","fer2013_mini_XCEPTION.102-0.66.hdf5"), compile=False)


model.compile(
    optimizer=Adam(learning_rate=0.0001), # Use the correct argument learning_rate
    loss='categorical_crossentropy',
    metrics=['accuracy']

    )

# Define emotion labels (as per FER-2013 dataset)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# ========================= #
# ===== Main function ===== #
# ========================= #

FPS = 10
TARGET_SIZE = (48, 48)  # Standard size for all images

def start_app(AllowWebcam, WebcamNumber):
    print("app starting")
    if AllowWebcam:
        cap = cv2.VideoCapture(WebcamNumber)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        print("web cam started")

    while AllowWebcam:
        feedback_text = ""
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            break

        # Detect and crop face
        face_found, cropped_frame = crop_face(frame)
        if not face_found:
            feedback_text = "No face detected"

        # Convert to RGB for Gradio display
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to grayscale for processing
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

        # Ensure consistent size
        gray_frame_resized = cv2.resize(gray_frame, TARGET_SIZE)

        # Convert to PIL Image for processing
        gray_frame_pil = Image.fromarray(gray_frame_resized)

        # Process with LBP + kNN
        lbp_frame, knn_prediction = process_knn(gray_frame_pil)
        # Resize LBP output for display
        lbp_frame = lbp_frame.resize((192, 192), Image.Resampling.NEAREST)
        
        # Process with HOG + SVM
        hog_frame, hog_prediction = process_hog(gray_frame_pil)
        # Resize HOG output for display
        hog_frame = hog_frame.resize((192, 192), Image.Resampling.NEAREST)

        # Process with mini-Xception cnn
        # Convert PIL image to numpy array
        cnn_input = np.array(gray_frame_pil)
        # Resize to 64x64
        cnn_input = cv2.resize(cnn_input, (64, 64))
        # Normalize pixel values
        cnn_input = cnn_input.astype('float32') / 255.0
        # Add batch and channel dimensions
        cnn_input = np.expand_dims(cnn_input, axis=-1)  # Add channel dimension
        cnn_input = np.expand_dims(cnn_input, axis=0)   # Add batch dimension
        # Get prediction
        emotion_prediction = model.predict(cnn_input, verbose=0)
        max_index = np.argmax(emotion_prediction[0])
        cnn_prediction = emotion_labels[max_index]
        # Create larger display version of input
        gray_frame_large = gray_frame_pil.resize((192, 192), Image.Resampling.LANCZOS)

        yield display_frame, lbp_frame, knn_prediction, hog_frame, hog_prediction, gray_frame_large, cnn_prediction, feedback_text
        
    if AllowWebcam:
        cap.release()

# ----- Gradio App Layout ----- #

with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():

            with gr.Column():

                # Inputs component
                allow_button = gr.Checkbox(label="Allow recording from webcam", value=True)
                webcam_number = gr.Number(label="Enter the webcam number", value=1)

                # Start/Restart button
                start_button = gr.Button("Start/Restart")
            
            # Outputs component
            webcam = gr.Image(label="Webcam Feed")

        feedback = gr.Textbox(label="Info :", interactive=False)
        
        with gr.Row():
            with gr.Column():
                knn_video = gr.Image(label="LBP + kNN Feed")
                knn_prediction = gr.Textbox(label="Emotion LBP + kNN", interactive=False)
            with gr.Column():
                hog_video = gr.Image(label="HOG + SVM Feed")
                hog_prediction = gr.Textbox(label="Emotion HOG + SVM", interactive=False)
            with gr.Column():
                cnn = gr.Image(label="CNN Feed")
                cnn_prediction = gr.Textbox(label="Emotion mini-Xception CNN", interactive=False)
            




    start_button.click(
        fn=start_app,       # Function to call
        inputs=[allow_button, webcam_number],       # Inputs to the function
        outputs=[webcam, knn_video, knn_prediction, hog_video, hog_prediction, cnn, cnn_prediction, feedback]      # Outputs from the function
    )

# Launch the app
print("Starting Gradio app...")
demo.launch()