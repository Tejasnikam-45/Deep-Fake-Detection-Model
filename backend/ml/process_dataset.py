import os
from utils.data_processor import DataProcessor
from tqdm import tqdm

def process_celeb_df():
    # Initialize the processor
    processor = DataProcessor(output_size=(224, 224))
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    # Input directories
    celeb_df_dir = os.path.join(data_dir, 'celeb-df')
    celeb_real_dir = os.path.join(celeb_df_dir, 'Celeb-real')
    youtube_real_dir = os.path.join(celeb_df_dir, 'YouTube-real')
    celeb_synthesis_dir = os.path.join(celeb_df_dir, 'Celeb-synthesis')
    
    # Output directories
    processed_dir = os.path.join(data_dir, 'processed', 'celeb-df')
    real_output_dir = os.path.join(processed_dir, 'real')
    fake_output_dir = os.path.join(processed_dir, 'fake')
    
    # Create output directories
    os.makedirs(real_output_dir, exist_ok=True)
    os.makedirs(fake_output_dir, exist_ok=True)
    
    print("Processing Celeb-DF dataset...")
    
    # Process real videos from Celeb-real
    print("\nProcessing Celeb-real videos...")
    for video in tqdm(os.listdir(celeb_real_dir)):
        if video.endswith(('.mp4', '.avi')):
            video_path = os.path.join(celeb_real_dir, video)
            output_path = os.path.join(real_output_dir, f'celeb_{os.path.splitext(video)[0]}')
            processor.process_video(video_path, output_path)
    
    # Process real videos from YouTube-real
    print("\nProcessing YouTube-real videos...")
    for video in tqdm(os.listdir(youtube_real_dir)):
        if video.endswith(('.mp4', '.avi')):
            video_path = os.path.join(youtube_real_dir, video)
            output_path = os.path.join(real_output_dir, f'youtube_{os.path.splitext(video)[0]}')
            processor.process_video(video_path, output_path)
    
    # Process fake videos from Celeb-synthesis
    print("\nProcessing Celeb-synthesis videos...")
    for video in tqdm(os.listdir(celeb_synthesis_dir)):
        if video.endswith(('.mp4', '.avi')):
            video_path = os.path.join(celeb_synthesis_dir, video)
            output_path = os.path.join(fake_output_dir, f'fake_{os.path.splitext(video)[0]}')
            processor.process_video(video_path, output_path)
    
    print("\nDataset processing completed!")
    
    # Print statistics
    real_videos = len([d for d in os.listdir(real_output_dir) if os.path.isdir(os.path.join(real_output_dir, d))])
    fake_videos = len([d for d in os.listdir(fake_output_dir) if os.path.isdir(os.path.join(fake_output_dir, d))])
    
    print("\nProcessing Statistics:")
    print(f"Real videos processed: {real_videos}")
    print(f"Fake videos processed: {fake_videos}")
    print(f"Total videos processed: {real_videos + fake_videos}")

if __name__ == '__main__':
    process_celeb_df() 