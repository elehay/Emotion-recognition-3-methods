# ----- Imports -----
import gradio as gr # For the web app
import cv2 # For webcam access

from skimage.feature import local_binary_pattern, hog # For LBP and HOG
from sklearn.neighbors import KNeighborsClassifier # For KNN
from sklearn.svm import LinearSVC # For SVM
from sklearn.preprocessing import StandardScaler # For feature scaling
import kagglehub # For loading datasets
import os # For file paths
from PIL import Image # For opening images 
import numpy as np # For array conversion
from tqdm import tqdm # For progress bars

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
cell_size = (8, 8)     # 8x8 pixels per cell
block_size = (2, 2)    # 2x2 cells per block
orientations = 9       # 9 orientation bins

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
print("Fitting HOG + SVM...")
svm_classifier.fit(hog_features_scaled, hog_emotions)
print("Done")

def process_hog(image):
    
    img_array = np.array(image, dtype=np.float64)
    
    # Calculate HOG features
    features = hog(img_array,
                  orientations=orientations,
                  pixels_per_cell=cell_size,
                  cells_per_block=block_size,
                  block_norm='L2-Hys',
                  feature_vector=True,
                  visualize=True)
    
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
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            break
            
        # Convert to RGB for Gradio display
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to grayscale for processing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Ensure consistent size
        gray_frame_resized = cv2.resize(gray_frame, TARGET_SIZE)

        # Convert to PIL Image for processing
        gray_frame_pil = Image.fromarray(gray_frame_resized)

        # Process with LBP + kNN
        lbp_frame, knn_prediction = process_knn(gray_frame_pil)
        
        # Process with HOG + SVM
        hog_frame, hog_prediction = process_hog(gray_frame_pil)

        yield display_frame, lbp_frame, knn_prediction, hog_frame, hog_prediction, gray_frame_pil
        
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
        
        with gr.Row():
            with gr.Column():
                knn_video = gr.Image(label="LBP + kNN Feed")
                knn_prediction = gr.Textbox(label="Emotion LBP + kNN", interactive=False)
            with gr.Column():
                hog_video = gr.Image(label="HOG + SVM Feed")
                hog_prediction = gr.Textbox(label="Emotion HOG + SVM", interactive=False)
            cnn = gr.Image(label="CNN Feed")




    start_button.click(
        fn=start_app,       # Function to call
        inputs=[allow_button, webcam_number],       # Inputs to the function
        outputs=[webcam, knn_video, knn_prediction, hog_video, hog_prediction, cnn]      # Outputs from the function
    )

# Launch the app
print("Starting Gradio app...")
demo.launch()