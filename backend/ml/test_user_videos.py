import os
import logging
import torch
import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm
from test_model import DeepfakeDetector

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, "dataset")
real_videos_dir = os.path.join(dataset_dir, "real")
fake_videos_dir = os.path.join(dataset_dir, "fake")
models_dir = os.path.join(current_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# Initialize face detector
detector = MTCNN()

def extract_faces(video_path):
    """Extract faces from video frames using MTCNN."""
    faces = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logging.info(f"Processing video: {video_path} with {total_frames} frames")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every 5th frame
        if frame_count % 5 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_faces = detector.detect_faces(frame_rgb)
            
            for face in detected_faces:
                x, y, w, h = face['box']
                # Add padding to ensure we capture the full face
                x = max(0, x - int(w * 0.1))
                y = max(0, y - int(h * 0.1))
                w = min(frame.shape[1] - x, w + int(w * 0.2))
                h = min(frame.shape[0] - y, h + int(h * 0.2))
                
                face_img = frame_rgb[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (224, 224))
                face_img = face_img / 255.0
                faces.append(face_img)
        
        frame_count += 1
        
    cap.release()
    logging.info(f"Extracted {len(faces)} faces from {video_path}")
    return faces

def process_video(video_path, model):
    """Process a video file and make predictions."""
    logging.info(f"Processing video: {video_path}")
    
    # Extract faces from video
    faces = extract_faces(video_path)
    if not faces:
        logging.warning(f"No faces detected in {video_path}")
        return None, None
    
    # Convert faces to tensor
    faces_tensor = torch.FloatTensor(np.array(faces)).permute(0, 3, 1, 2)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(faces_tensor)
        probabilities = torch.softmax(predictions, dim=1)
        
    # Get average prediction
    avg_prob = probabilities.mean(dim=0)
    prediction = avg_prob.argmax().item()
    confidence = avg_prob[prediction].item()
    
    # Apply calibration
    calibrated_prob = torch.softmax(predictions / model.temperature, dim=1)
    calibrated_avg_prob = calibrated_prob.mean(dim=0)
    calibrated_confidence = calibrated_avg_prob[prediction].item()
    
    logging.info(f"Prediction: {'Fake' if prediction == 1 else 'Real'}")
    logging.info(f"Raw Confidence: {confidence:.4f}")
    logging.info(f"Calibrated Confidence: {calibrated_confidence:.4f}")
    
    return prediction, calibrated_confidence

def main():
    # Load model
    model_path = os.path.join(models_dir, "best_model.pth")
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return
    
    logging.info("Loading model...")
    model = DeepfakeDetector()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    logging.info("Model loaded successfully.")
    
    # Process real videos
    logging.info("Processing real videos...")
    real_results = []
    for video_file in os.listdir(real_videos_dir):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(real_videos_dir, video_file)
            prediction, confidence = process_video(video_path, model)
            if prediction is not None:
                real_results.append((video_file, prediction, confidence))
    
    # Process fake videos
    logging.info("Processing fake videos...")
    fake_results = []
    for video_file in os.listdir(fake_videos_dir):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(fake_videos_dir, video_file)
            prediction, confidence = process_video(video_path, model)
            if prediction is not None:
                fake_results.append((video_file, prediction, confidence))
    
    # Print results
    logging.info("\nResults Summary:")
    logging.info("Real Videos:")
    for video_file, prediction, confidence in real_results:
        logging.info(f"  {video_file}: {'Fake' if prediction == 1 else 'Real'} (Confidence: {confidence:.4f})")
    
    logging.info("\nFake Videos:")
    for video_file, prediction, confidence in fake_results:
        logging.info(f"  {video_file}: {'Fake' if prediction == 1 else 'Real'} (Confidence: {confidence:.4f})")
    
    # Calculate accuracy
    real_correct = sum(1 for _, pred, _ in real_results if pred == 0)
    fake_correct = sum(1 for _, pred, _ in fake_results if pred == 1)
    total_correct = real_correct + fake_correct
    total_videos = len(real_results) + len(fake_results)
    
    if total_videos > 0:
        accuracy = total_correct / total_videos
        logging.info(f"\nAccuracy: {accuracy:.2f} ({total_correct}/{total_videos})")

if __name__ == "__main__":
    main() 