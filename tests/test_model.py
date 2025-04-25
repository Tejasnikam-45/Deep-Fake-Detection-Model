import cv2
import numpy as np
import torch
import torch.nn as nn
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torchvision import transforms
import argparse
import logging
from pathlib import Path
import shutil
from datetime import datetime
from model import DeepfakeDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")

# Initialize MTCNN for face detection
mtcnn = MTCNN()

def extract_faces(video_path):
    """
    Extract faces from video using MTCNN.
    """
    logging.info(f"Extracting faces from {video_path}")
    faces = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error(f"Error opening video file {video_path}")
        return []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = frame_rgb[y1:y2, x1:x2]
                if face.size > 0:
                    faces.append(cv2.resize(face, (224, 224)))  # Resize to match model input

    cap.release()
    return faces

def save_results(video_path, faces, predictions, output_dir):
    """
    Save face images and predictions to the output directory.
    """
    video_name = os.path.basename(video_path)
    video_result_dir = os.path.join(output_dir, video_name.split(".")[0])
    os.makedirs(video_result_dir, exist_ok=True)

    for i, (face, score) in enumerate(zip(faces, predictions)):
        plt.imsave(os.path.join(video_result_dir, f"face_{i}.jpg"), face)
        with open(os.path.join(video_result_dir, "predictions.txt"), "a") as f:
            f.write(f"Face {i}: Score={score:.4f}\n")

def process_video(video_path, output_dir, model, device):
    """
    Process a video file to detect deepfakes.
    """
    logging.info(f"Processing video: {video_path}")
    faces = extract_faces(video_path)

    if not faces:
        logging.error(f"No faces extracted from {video_path}")
        return None, None

    predictions = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for face in faces:
        face_tensor = transform(face).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face_tensor)
            probs = torch.softmax(output, dim=1)
            fake_prob = probs[0, 1].item()
            predictions.append(fake_prob)

    if predictions:
        avg_prediction = sum(predictions) / len(predictions)
        logging.info(f"Average prediction for {video_path}: {avg_prediction:.4f}")
        save_results(video_path, faces, predictions, output_dir)

        verdict = "FAKE" if avg_prediction > 0.5 else "REAL"
        with open(os.path.join(output_dir, "overall_results.txt"), "a") as f:
            f.write(f"{os.path.basename(video_path)}: Score={avg_prediction:.3f}, Verdict={verdict}\n")

        logging.info(f"Video processed - Score: {avg_prediction:.3f}, Verdict: {verdict}")
        return avg_prediction, faces
    else:
        logging.warning("No predictions made")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Test deepfake detection model on videos")
    parser.add_argument("--model", type=str, default="backend/ml/models/best_model.pth", help="Path to the trained model")
    parser.add_argument("--videos", type=str, default="test_videos", help="Path to directory containing test videos")
    parser.add_argument("--output", type=str, default="test_results", help="Path to save test results")
    args = parser.parse_args()

    logging.info("Deepfake Detection System")
    logging.info("------------------------")

    # Ensure the test videos directory exists
    if not os.path.exists(args.videos):
        logging.error(f"Video directory '{args.videos}' not found!")
        exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Clean the output directory
    for item in os.listdir(args.output):
        item_path = os.path.join(args.output, item)
        try:
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)
        except Exception as e:
            logging.error(f"Error cleaning up {item_path}: {str(e)}")

    with open(os.path.join(args.output, "overall_results.txt"), "w") as f:
        f.write("Overall Results:\n")

    model_path = os.path.abspath(args.model)
    logging.info(f"Looking for model at: {model_path}")
    logging.info("Loading model...")

    model = DeepfakeDetector().to(DEVICE)

    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        logging.error(f"Model loading error: {str(e)}")
        exit(1)

    model.eval()
    logging.info("Model loaded successfully")

    for file_name in os.listdir(args.videos):
        if file_name.endswith((".mp4", ".avi", ".mov")):
            logging.info(f"\nProcessing video: {file_name}")
            file_path = os.path.join(args.videos, file_name)
            process_video(str(file_path), args.output, model, DEVICE)

if __name__ == "__main__":
    main()
