import os
import shutil
from pathlib import Path
from tqdm import tqdm
from data_processor import DataProcessor

def organize_celeb_df(source_dir: str, output_dir: str):
    """
    Organize Celeb-DF dataset into real and fake categories.
    Args:
        source_dir: Path to the celeb-df directory
        output_dir: Path to save processed data
    """
    # Create output directories
    real_dir = os.path.join(output_dir, 'real')
    fake_dir = os.path.join(output_dir, 'fake')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # Initialize data processor
    processor = DataProcessor()

    # Process Celeb-real videos (real)
    celeb_real_dir = os.path.join(source_dir, 'Celeb-real')
    if os.path.exists(celeb_real_dir):
        print("Processing Celeb-real videos...")
        for video_file in tqdm(os.listdir(celeb_real_dir)):
            if video_file.endswith(('.mp4', '.avi')):
                video_path = os.path.join(celeb_real_dir, video_file)
                output_path = os.path.join(real_dir, 'celeb_' + video_file.split('.')[0])
                processor.process_video(video_path, output_path)

    # Process YouTube-real videos (real)
    youtube_real_dir = os.path.join(source_dir, 'YouTube-real')
    if os.path.exists(youtube_real_dir):
        print("Processing YouTube-real videos...")
        for video_file in tqdm(os.listdir(youtube_real_dir)):
            if video_file.endswith(('.mp4', '.avi')):
                video_path = os.path.join(youtube_real_dir, video_file)
                output_path = os.path.join(real_dir, 'youtube_' + video_file.split('.')[0])
                processor.process_video(video_path, output_path)

    # Process Celeb-synthesis videos (fake)
    celeb_synthesis_dir = os.path.join(source_dir, 'Celeb-synthesis')
    if os.path.exists(celeb_synthesis_dir):
        print("Processing Celeb-synthesis videos...")
        for video_file in tqdm(os.listdir(celeb_synthesis_dir)):
            if video_file.endswith(('.mp4', '.avi')):
                video_path = os.path.join(celeb_synthesis_dir, video_file)
                output_path = os.path.join(fake_dir, video_file.split('.')[0])
                processor.process_video(video_path, output_path)

    print(f"Dataset processing completed. Processed data saved to {output_dir}")

if __name__ == '__main__':
    # Update these paths according to your directory structure
    celeb_df_dir = 'backend/ml/data/celeb-df'
    output_dir = 'backend/ml/data/processed/celeb-df'
    
    organize_celeb_df(celeb_df_dir, output_dir) 