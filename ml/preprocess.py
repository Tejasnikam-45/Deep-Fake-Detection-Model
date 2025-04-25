import cv2

import os

from mtcnn import MTCNN

import numpy as np

from sklearn.model_selection import train_test_split

import albumentations as A

from tqdm import tqdm

import logging

# Set up logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths for both datasets

ff_real_path = "faceforensics/FF++/real"

ff_fake_path = "faceforensics/FF++/fake"

celeb_real_path = "celeb_df/Celeb-real"

celeb_fake_path = "celeb_df/Celeb-synthesis"

output_path = "processed_data_expanded"

# Check if directories exist

for path in [ff_real_path, ff_fake_path, celeb_real_path, celeb_fake_path]:

  if not os.path.exists(path):

    logging.error(f"Directory not found: {path}")

    raise FileNotFoundError(f"Directory not found: {path}")

  if not os.listdir(path):

    logging.error(f"Directory is empty: {path}")

    raise ValueError(f"Directory is empty: {path}")

os.makedirs(output_path, exist_ok=True)

logging.info("Output directory created/verified")

# Initialize face detector

try:

  detector = MTCNN()

  logging.info("MTCNN detector initialized successfully")

except Exception as e:

  logging.error(f"Failed to initialize MTCNN detector: {str(e)}")

  raise

# Define augmentation pipeline

aug_pipeline = A.Compose([

  A.RandomBrightnessContrast(p=0.5),

  A.HorizontalFlip(p=0.5),

  A.RandomRotate90(p=0.5),

  A.GaussNoise(p=0.3),

  A.ElasticTransform(p=0.3),

])

logging.info("Augmentation pipeline defined")

def extract_faces(video_path, label, max_frames=15):

  if not os.path.exists(video_path):

    logging.error(f"Video file not found: {video_path}")

    return [], []

  try:

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

      logging.error(f"Failed to open video: {video_path}")

      return [], []

    faces = []

    labels = []

    count = 0

    while cap.isOpened() and count < max_frames:

      ret, frame = cap.read()

      if not ret:

        break

      # Convert to RGB for MTCNN

      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      result = detector.detect_faces(frame_rgb)

      if result:

        # Get the largest face

        face_box = max(result, key=lambda x: x['box'][2] * x['box'][3])

        x, y, w, h = face_box['box']

        # Add padding

        padding = 20

        y1 = max(0, y - padding)

        y2 = min(frame.shape[0], y + h + padding)

        x1 = max(0, x - padding)

        x2 = min(frame.shape[1], x + w + padding)

        face = frame[y1:y2, x1:x2]

        face = cv2.resize(face, (224, 224))

        face = face.astype(np.float32) / 255.0 # Convert to float32

        try:

          # Apply augmentations

          augmented = aug_pipeline(image=face)

          face_aug = augmented['image']

          faces.append(face)

          faces.append(face_aug) # Add augmented version

          labels.extend([label, label])

          count += 1

        except Exception as e:

          logging.warning(f"Failed to apply augmentation to face from {video_path}: {str(e)}")

          continue

    cap.release()

    logging.debug(f"Processed video {video_path}: extracted {len(faces)} faces")

    return faces, labels

  except Exception as e:

    logging.error(f"Error processing video {video_path}: {str(e)}")

    return [], []

def process_dataset(real_path, fake_path, max_videos=100):

  real_faces, real_labels = [], []

  fake_faces, fake_labels = [], []

  # Process real videos

  logging.info(f"Processing real videos from {real_path}")

  real_videos = [v for v in os.listdir(real_path) if v.endswith(('.mp4', '.avi'))]

  if not real_videos:

    logging.error(f"No video files found in {real_path}")

    raise ValueError(f"No video files found in {real_path}")

  for video in tqdm(real_videos[:max_videos]):

    faces, labels = extract_faces(os.path.join(real_path, video), 0)

    if faces: # Only add if faces were successfully extracted

      real_faces.extend(faces)

      real_labels.extend(labels)

  # Process fake videos

  logging.info(f"Processing fake videos from {fake_path}")

  fake_videos = [v for v in os.listdir(fake_path) if v.endswith(('.mp4', '.avi'))]

  if not fake_videos:

    logging.error(f"No video files found in {fake_path}")

    raise ValueError(f"No video files found in {fake_path}")

  for video in tqdm(fake_videos[:max_videos]):

    faces, labels = extract_faces(os.path.join(fake_path, video), 1)

    if faces: # Only add if faces were successfully extracted

      fake_faces.extend(faces)

      fake_labels.extend(labels)

  if not real_faces or not fake_faces:

    logging.error(f"No faces extracted from dataset: {real_path}, {fake_path}")

    raise ValueError(f"No faces extracted from dataset: {real_path}, {fake_path}")

  return real_faces, real_labels, fake_faces, fake_labels

try:

  # Process both datasets

  logging.info("Processing FaceForensics++ dataset...")

  ff_real_faces, ff_real_labels, ff_fake_faces, ff_fake_labels = process_dataset(ff_real_path, ff_fake_path)

  logging.info("Processing Celeb-DF dataset...")

  celeb_real_faces, celeb_real_labels, celeb_fake_faces, celeb_fake_labels = process_dataset(celeb_real_path, celeb_fake_path)

  # Combine datasets and ensure correct data type

  X = np.array(ff_real_faces + ff_fake_faces + celeb_real_faces + celeb_fake_faces, dtype=np.float32)

  y = np.array(ff_real_labels + ff_fake_labels + celeb_real_labels + celeb_fake_labels, dtype=np.int32)

  if len(X) == 0:

    raise ValueError("No data was processed successfully")

  # Split into train, validation, and test sets

  X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

  X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

  # Save processed data

  logging.info("Saving processed data...")

  np.save(os.path.join(output_path, "X_train.npy"), X_train)

  np.save(os.path.join(output_path, "y_train.npy"), y_train)

  np.save(os.path.join(output_path, "X_val.npy"), X_val)

  np.save(os.path.join(output_path, "y_val.npy"), y_val)

  np.save(os.path.join(output_path, "X_test.npy"), X_test)

  np.save(os.path.join(output_path, "y_test.npy"), y_test)

  logging.info(f"\nDataset Statistics:")

  logging.info(f"Total samples: {len(X)}")

  logging.info(f"Training set: {len(X_train)} ({len(X_train[y_train==0])} real, {len(X_train[y_train==1])} fake)")

  logging.info(f"Validation set: {len(X_val)} ({len(X_val[y_val==0])} real, {len(X_val[y_val==1])} fake)")

  logging.info(f"Test set: {len(X_test)} ({len(X_test[y_test==0])} real, {len(X_test[y_test==1])} fake)")

except Exception as e:

  logging.error(f"An error occurred during processing: {str(e)}")

  raise

































































