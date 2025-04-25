import cv2
import numpy as np
import os
from tqdm import tqdm
from typing import Tuple, List

class DataProcessor:
    def __init__(self, output_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the DataProcessor.
        Args:
            output_size: Tuple of (height, width) for output frames
        """
        self.output_size = output_size
        # Load the face cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def extract_frames(self, video_path: str, output_dir: str, frames_per_second: int = 1) -> List[str]:
        """
        Extract frames from a video file.
        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted frames
            frames_per_second: Number of frames to extract per second
        Returns:
            List of paths to extracted frames
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / frames_per_second)
        frame_paths = []
        frame_count = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f'frame_{frame_count:06d}.jpg')
                frame = cv2.resize(frame, self.output_size)
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)

            frame_count += 1

        video.release()
        return frame_paths

    def detect_and_align_face(self, image_path: str) -> np.ndarray:
        """
        Detect and align face in an image using OpenCV.
        Args:
            image_path: Path to the input image
        Returns:
            Aligned face image as numpy array
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None

        # Get the largest face
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        
        x, y, w, h = faces[0]
        
        # Add margin around the face
        margin = int(max(w, h) * 0.1)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        # Extract and resize the face
        face_image = image[y:y+h, x:x+w]
        face_image = cv2.resize(face_image, self.output_size)
        
        return face_image

    def process_dataset(self, dataset_dir: str, output_dir: str):
        """
        Process entire dataset.
        Args:
            dataset_dir: Directory containing the dataset
            output_dir: Directory to save processed data
        """
        for root, _, files in os.walk(dataset_dir):
            for file in tqdm(files):
                if file.endswith(('.mp4', '.avi')):
                    video_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, dataset_dir)
                    frames_dir = os.path.join(output_dir, relative_path, os.path.splitext(file)[0])
                    
                    # Extract frames
                    frame_paths = self.extract_frames(video_path, frames_dir)
                    
                    # Process each frame
                    for frame_path in frame_paths:
                        aligned_face = self.detect_and_align_face(frame_path)
                        if aligned_face is not None:
                            cv2.imwrite(frame_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
                        else:
                            os.remove(frame_path)  # Remove frames where no face is detected

    def process_video(self, video_path: str, output_dir: str) -> None:
        """
        Process a single video file.
        Args:
            video_path: Path to the video file
            output_dir: Directory to save processed frames
        """
        # Extract frames
        frame_paths = self.extract_frames(video_path, output_dir)
        
        # Process each frame
        for frame_path in frame_paths:
            aligned_face = self.detect_and_align_face(frame_path)
            if aligned_face is not None:
                cv2.imwrite(frame_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
            else:
                os.remove(frame_path)  # Remove frames where no face is detected

if __name__ == '__main__':
    processor = DataProcessor()
    
    # Process FaceForensics++ dataset
    ff_dataset_dir = 'path/to/faceforensics++'
    ff_output_dir = 'processed/faceforensics++'
    processor.process_dataset(ff_dataset_dir, ff_output_dir)
    
    # Process Celeb-DF dataset
    celeb_dataset_dir = 'path/to/celeb-df'
    celeb_output_dir = 'processed/celeb-df'
    processor.process_dataset(celeb_dataset_dir, celeb_output_dir) 