import subprocess

# Replace this with your actual YouTube link
youtube_link = "https://www.youtube.com/watch?v=u4UZ4UvZXrg"

# Function to extract video stream URL using yt-dlp
def get_stream_url(youtube_link):
    command = ["yt-dlp", "-f", "best", "-g", youtube_link]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        print("Error extracting stream URL:", result.stderr)
        return None

# Get the stream URL
stream_url = get_stream_url(youtube_link)

if stream_url:
    print(f"Playing stream from: {stream_url}")
    # Use ffplay to play the video
    subprocess.run(["ffplay", stream_url])
else:
    print("Failed to retrieve the stream URL.")
