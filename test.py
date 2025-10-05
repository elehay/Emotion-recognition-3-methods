# ----- Imports -----
import gradio as gr # For the web app
import cv2 # For webcam access

from skimage.feature import local_binary_pattern # For LBP
from sklearn.neighbors import KNeighborsClassifier # For KNN
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
for emotion_directory in tqdm(os.listdir(training_path), desc="kNN Training"):
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


# ========================= #
# ===== Main function ===== #
# ========================= #

FPS = 10
WIDTH = 48
HEIGHT = 48

def start_app(AllowWebcam, WebcamNumber=0):
    if AllowWebcam:
        cap = cv2.VideoCapture(WebcamNumber)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    while AllowWebcam:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            break
            
        # Convert to RGB for Gradio display
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to grayscale for processing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Process with LBP + kNN
        lbp_frame, knn_prediction = process_knn(gray_frame)

        yield display_frame, lbp_frame, knn_prediction, gray_frame, gray_frame
        
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
                knn_prediction = gr.Textbox(label="Emotion", interactive=False)
            hog = gr.Image(label="HOG + SVM Feed")
            cnn = gr.Image(label="CNN Feed")




    start_button.click(
        fn=start_app,       # Function to call
        inputs=[allow_button, webcam_number],       # Inputs to the function
        outputs=[webcam, knn_video, knn_prediction, hog, cnn]      # Outputs from the function
    )

# Launch the app
demo.launch()