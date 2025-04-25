import cv2
import os
import numpy as np
import logging
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths for datasets
DATASETS = {
    "ff_real": "faceforensics/FF++/real",
    "ff_fake": "faceforensics/FF++/fake",
    "celeb_real": "celeb_df/Celeb-real",
    "celeb_fake": "celeb_df/Celeb-synthesis",
}
OUTPUT_PATH = "processed_data_expanded"

# Verify dataset directories
for name, path in DATASETS.items():
    if not os.path.exists(path) or not os.listdir(path):
        logging.error(f"Dataset missing or empty: {path}")
        raise FileNotFoundError(f"Dataset missing or empty: {path}")

os.makedirs(OUTPUT_PATH, exist_ok=True)
logging.info("Output directory verified.")

# Initialize face detector
try:
    detector = MTCNN()
    logging.info("MTCNN detector initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize MTCNN detector: {e}")
    raise

# Define augmentation pipeline
AUGMENTATIONS = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.GaussNoise(p=0.3),
    A.ElasticTransform(p=0.3),
])

logging.info("Augmentation pipeline defined.")

def extract_faces(video_path, label, max_frames=15):
    """Extracts faces from a video and applies augmentations."""
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return [], []
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            return [], []
        
        faces, labels = [], []
        count = 0
        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(frame_rgb)
            
            if results:
                face_box = max(results, key=lambda x: x['box'][2] * x['box'][3])
                x, y, w, h = face_box['box']
                padding = 20
                y1, y2 = max(0, y - padding), min(frame.shape[0], y + h + padding)
                x1, x2 = max(0, x - padding), min(frame.shape[1], x + w + padding)
                
                face = frame[y1:y2, x1:x2]
                face = cv2.resize(face, (224, 224)).astype(np.float32) / 255.0
                
                try:
                    augmented = AUGMENTATIONS(image=face)['image']
                    faces.extend([face, augmented])
                    labels.extend([label, label])
                    count += 1
                except Exception as e:
                    logging.warning(f"Failed to augment face from {video_path}: {e}")
                    continue
        
        cap.release()
        return faces, labels
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {e}")
        return [], []

def process_dataset(real_path, fake_path, max_videos=100):
    """Processes a dataset by extracting faces from videos."""
    real_faces, real_labels, fake_faces, fake_labels = [], [], [], []
    
    for category, path in [("real", real_path), ("fake", fake_path)]:
        logging.info(f"Processing {category} videos from {path}")
        videos = [v for v in os.listdir(path) if v.endswith(('.mp4', '.avi'))][:max_videos]
        if not videos:
            raise ValueError(f"No video files found in {path}")
        
        for video in tqdm(videos):
            faces, labels = extract_faces(os.path.join(path, video), 0 if category == "real" else 1)
            if category == "real":
                real_faces.extend(faces)
                real_labels.extend(labels)
            else:
                fake_faces.extend(faces)
                fake_labels.extend(labels)
    
    if not real_faces or not fake_faces:
        raise ValueError(f"No faces extracted from dataset: {real_path}, {fake_path}")
    
    return real_faces, real_labels, fake_faces, fake_labels

try:
    logging.info("Processing datasets...")
    ff_real_faces, ff_real_labels, ff_fake_faces, ff_fake_labels = process_dataset(DATASETS['ff_real'], DATASETS['ff_fake'])
    celeb_real_faces, celeb_real_labels, celeb_fake_faces, celeb_fake_labels = process_dataset(DATASETS['celeb_real'], DATASETS['celeb_fake'])

    X = np.array(ff_real_faces + ff_fake_faces + celeb_real_faces + celeb_fake_faces, dtype=np.float32)
    y = np.array(ff_real_labels + ff_fake_labels + celeb_real_labels + celeb_fake_labels, dtype=np.int32)

    if len(X) == 0:
        raise ValueError("No data was processed successfully")
    
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Save processed data
    logging.info("Saving processed data...")
    for name, data in zip(["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"], [X_train, y_train, X_val, y_val, X_test, y_test]):
        np.save(os.path.join(OUTPUT_PATH, f"{name}.npy"), data)
    
    logging.info("Dataset successfully processed and saved.")
    logging.info(f"Training set: {len(X_train)} samples")
    logging.info(f"Validation set: {len(X_val)} samples")
    logging.info(f"Test set: {len(X_test)} samples")
except Exception as e:
    logging.error(f"An error occurred during processing: {e}")
    raise
