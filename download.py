from pytube import YouTube
import os

def download_youtube_video(url, output_path='.'):
    try:
        # Create a YouTube object with the URL
        yt = YouTube(url)

        # Get the highest resolution stream available
        stream = yt.streams.get_highest_resolution()

        print(f"Downloading '{yt.title}'...")

        # Download the video to the specified output path
        stream.download(output_path=output_path)

        print(f"Download completed!\nVideo saved to: {output_path}/{yt.title}.mp4")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example YouTube video URL
    url = input("Enter the YouTube video URL: ")
    output_path = input("Enter the directory to save the video (leave blank for current directory): ") or '.'

    download_youtube_video(url, output_path)
