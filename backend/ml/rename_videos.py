import os
import shutil
from tqdm import tqdm

def rename_videos(input_dir=None):
    """
    Rename videos with appropriate prefixes.
    Args:
        input_dir: Optional custom input directory path
    """
    if input_dir:
        celeb_df_dir = input_dir
    else:
        # Define paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data')
        celeb_df_dir = os.path.join(data_dir, 'celeb-df')
    
    if not os.path.exists(celeb_df_dir):
        print(f"Error: Directory not found: {celeb_df_dir}")
        print("Please provide the correct path to the Celeb-DF dataset directory.")
        return

    # Create backup directory
    backup_dir = os.path.join(os.path.dirname(celeb_df_dir), 'celeb-df_backup')
    
    try:
        # Backup original files
        print("Creating backup of original files...")
        if not os.path.exists(backup_dir):
            shutil.copytree(celeb_df_dir, backup_dir)
            print(f"Backup created at: {backup_dir}")
        else:
            print("Backup already exists, skipping backup creation.")
        
        # Rename real videos
        real_dir = os.path.join(celeb_df_dir, 'real')
        if os.path.exists(real_dir):
            print("\nRenaming real videos...")
            videos = [f for f in os.listdir(real_dir) if f.endswith(('.mp4', '.avi'))]
            for video in tqdm(videos):
                if not (video.startswith('celeb_') or video.startswith('youtube_')):
                    old_path = os.path.join(real_dir, video)
                    # Determine if it's a YouTube video based on filename pattern
                    if 'youtube' in video.lower():
                        new_name = f"youtube_{video}"
                    else:
                        new_name = f"celeb_{video}"
                    new_path = os.path.join(real_dir, new_name)
                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        print(f"Renamed: {video} -> {new_name}")

        # Rename fake videos
        fake_dir = os.path.join(celeb_df_dir, 'fake')
        if os.path.exists(fake_dir):
            print("\nRenaming fake videos...")
            videos = [f for f in os.listdir(fake_dir) if f.endswith(('.mp4', '.avi'))]
            for video in tqdm(videos):
                if not video.startswith('fake_'):
                    old_path = os.path.join(fake_dir, video)
                    new_name = f"fake_{video}"
                    new_path = os.path.join(fake_dir, new_name)
                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        print(f"Renamed: {video} -> {new_name}")

        print("\nRenaming completed!")
        
        # Print statistics for directories that exist
        print("\nStatistics:")
        if os.path.exists(real_dir):
            celeb_count = len([f for f in os.listdir(real_dir) if f.startswith('celeb_')])
            youtube_count = len([f for f in os.listdir(real_dir) if f.startswith('youtube_')])
            print(f"Real videos: {celeb_count} celeb files, {youtube_count} youtube files")
        if os.path.exists(fake_dir):
            print(f"Fake videos: {len([f for f in os.listdir(fake_dir) if f.startswith('fake_')])} files")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Please check if you have the correct permissions and the directories exist.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        rename_videos(sys.argv[1])
    else:
        rename_videos() 