import os
import urllib.request
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def download_videos():
    # Create test_videos directory if it doesn't exist
    os.makedirs('test_videos', exist_ok=True)
    
    # Sample videos (these are publicly available test videos)
    videos = {
        'celeb_id9_0000.mp4': 'https://github.com/ondyari/FaceForensics/raw/master/dataset_samples/original_sequences/actors/c23/videos/000.mp4',
        'fake_00200.mp4': 'https://github.com/ondyari/FaceForensics/raw/master/dataset_samples/manipulated_sequences/Deepfakes/c23/videos/000_003.mp4'
    }
    
    for filename, url in videos.items():
        output_path = os.path.join('test_videos', filename)
        if not os.path.exists(output_path):
            logging.info(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, output_path)
                logging.info(f"Successfully downloaded {filename}")
            except Exception as e:
                logging.error(f"Failed to download {filename}: {str(e)}")
        else:
            logging.info(f"{filename} already exists, skipping download")

if __name__ == "__main__":
    download_videos() 