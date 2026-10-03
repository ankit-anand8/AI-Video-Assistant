# Purpose: Sends audio chunks to Groq's hosted Whisper model and gets text back
# What this file does:
# Creates the Groq client once and reuses it (no model download, no local CPU work)
# Transcribes each audio chunk one by one
# Supports translate mode - convert any speech directly into English text
# Combines all chunk transcripts into one single clean transcript
# Model name is configurable via .env
#
# Needs:  pip install groq
# .env:   GROQ_API_KEY=your_key_here
#         WHISPER_MODEL=whisper-large-v3

import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3")  # model name on Groq's servers
groq_client = None  # Initially, no client is created


def get_client():
    global groq_client
    # Allows the function to modify the global groq_client  variable, so the client stays available outside
    if groq_client is None:   #Checks whether the Groq client has already been created.
        print("Creating Groq client ....")
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("Groq client ready")
    return groq_client
# First call -> create client -> store it -> return it
# Next calls -> client already exists -> return the same one


def transcribe_chunk(audio_file, translate=False):  #Input: one audio file --> Returns:text
#It means: "Take chunk1.wav, send it to Whisper, and give me its text."
    
    client = get_client()  #It gives you the Groq client, which is the object we use to communicate with Groq's API.

    #The with also makes sure the file is properly closed afterward.
    with open(audio_file, "rb") as file: #The "rb" means: r = read, b = binary . Audio is binary data, so we read it as binary.
        audio = (os.path.basename(audio_file), file.read()) #"Make a package containing the filename and the actual audio data."

    if translate:
        result = client.audio.translations.create(
            file=audio,
            model="whisper-large-v3",
            response_format="json",
            temperature=0.0
        )
    else:
        result = client.audio.transcriptions.create(
            file=audio,
            model=WHISPER_MODEL,
            response_format="json",
            temperature=0.0
        )

    return result.text


def transcribe_all(audio_chunks: list[str], translate: bool = False) -> str:
    #audio_chunks is basically the list returned by process_input() i.e from the audio_processor.py
    
    all_text = []
    for chunk in audio_chunks:
        text = transcribe_chunk(chunk, translate)
        all_text.append(text.strip()) #strip() removes unnecessary spaces and newline characters from the beginning and end of the text.

    return " ".join(all_text) #Returns a single string (str) containing all the chunk transcripts joined together.