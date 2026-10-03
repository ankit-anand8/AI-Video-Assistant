#Purpose: Accept any audio/video source and convert in into WAV (Waveform Audio File Format) chunks ready for Whisper
#What will this file do:
#Detect if input is a YouTube URL or a local file
#Download audio-only stream from YouTube using yt-dlp
#Convert any format (MP3,MP4,M4A) to WAV using ffmpeg . WAV is a file format, not an audio-processing library or a model
#Splits large audio files into 1-minute chunks for Whisper
#Returns a list of chunk file paths to the next step

import yt_dlp #lets your program take a YouTube URL and download its audio, which can then be passed to Whisper for transcription.
from pydub import AudioSegment #AudioSegment is used to load, edit, and convert audio files.

import os

DOWNLOAD_DIR="downloads" #Save the downloaded files inside the "downloads" folder.
os.makedirs(DOWNLOAD_DIR,exist_ok=True) #os.makedirs() creates a directory/folder. Here it makes the "downloads" folder
#exist_ok=True mean?
#It means: If the folder already exists, don't give an error.
#Without it: os.makedirs("downloads")
#If downloads already exists, Python would raise an error.

#Downloading from Youtube (Use GitHub for this)
def download_youtube_audio(url: str) -> str:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"),  # id = safe filename
        "noplaylist": True,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info) 


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format"""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"  #Creates the output file path, changing the extension to .wav
    audio = AudioSegment.from_file(input_path) #Loads the input audio/video file using Pydub.
    audio = audio.set_channels(1).set_frame_rate(16000) #Converts audio to mono (1 channel) and 16 kHz.
    audio.export(output_path, format="wav") #Saves/exports the processed audio as a WAV file.
    return output_path


# This function takes an audio or video file and converts it into a standardized WAV format for Whisper. 
# It loads the file using Pydub, converts the audio to mono (1 channel) because speech generally doesn't need separate left and right
# channels, and resamples it to 16 kHz. We use 16 kHz because Whisper is designed to work with audio sampled at 16,000 samples per second,
# which is sufficient to capture the important frequencies of human speech while keeping processing efficient. 
# It then saves the audio as a WAV file and returns the path of the converted file.



#Next step: Chunking a video into chunks of 1 minute. 
def chunk_audio(wav_path: str, chunk_minutes: int = 1) -> list:
    audio = AudioSegment.from_wav(wav_path)  # Open the WAV file located at wav_path and load its audio data into audio.
    #.from_wav() → tells Pydub that the input file is a WAV file and to load it.
    chunk_ms = chunk_minutes * 60 * 1000 #convert the chunk in minute to chunk in milliseconds
    chunks_list = []
    base = os.path.splitext(wav_path)[0]

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = f"{base}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks_list.append(chunk_path)
    return chunks_list

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"): # if source.startswith(("http://", "https://")):
        print("Detected YouTube URL. Downloading the audio....")
        downloaded_path = download_youtube_audio(source)
        # Standardize YouTube audio
        wav_path = convert_to_wav(downloaded_path)

    else:
        print("Detected local file. Converting to WAV....")
        wav_path = convert_to_wav(source)
    print("Chunking audio....")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created")
    return chunks

#For this project:
#- YouTube downloading → use/copy the GitHub helper code.
#- Local audio/video conversion → use/copy the helper code.
#- Chunking logic → write it yourself.
#- Choosing whether the input is YouTube or a local file → write it yourself.