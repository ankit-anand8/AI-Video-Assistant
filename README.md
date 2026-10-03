# 🎥 AI Video Assistant

A Generative AI RAG application that allows users to provide a YouTube URL or local audio/video file and ask questions about its content.

## ✨ Features

- 🎥 Supports YouTube URLs
- 🎵 Supports local audio/video files
- 🎙️ Converts speech to text using Whisper
- ✂️ Splits long audio into smaller chunks
- 🔢 Creates embeddings using Gemini
- 🗄️ Stores transcript vectors using Chroma
- 🔍 Uses MMR-based retrieval
- 🤖 Generates answers using Google Gemini
- 📋 Supports video summaries
- 💬 Interactive interface using Streamlit
- 📚 Answers questions based only on the video content

## 🛠️ Technologies Used

- Python
- LangChain
- Google Gemini
- Groq Whisper
- Chroma
- Streamlit
- yt-dlp
- PyDub
- FFmpeg
- RecursiveCharacterTextSplitter

## 🔄 How It Works

The application accepts a YouTube URL or a local audio/video file.

The audio is extracted, converted into WAV format, and divided into smaller chunks.

Each audio chunk is transcribed using Groq's hosted Whisper model. The transcripts are then combined into a complete transcript.

The transcript is split into smaller chunks and converted into embeddings using Gemini. These embeddings are stored in Chroma.

When the user asks a question, relevant transcript chunks are retrieved using MMR and provided to Gemini as context.

Gemini then generates an answer based only on the video transcript.

## 🌐 Links

- 🚀 Live Demo: **[Link will be added]**
- 💻 GitHub Repository: **[Link will be added]**

## 📁 Project Files

- `app.py` — Streamlit application and user interface
- `main.py` — Main RAG workflow and question-answering logic
- `core/transcriber.py` — Audio transcription using Groq Whisper
- `core/database.py` — Transcript chunking, embeddings, and Chroma vector database
- `utils/audio_processor.py` — YouTube downloading, audio conversion, and audio chunking

## 🎯 Purpose

This project was built to practice and strengthen Generative AI and RAG concepts, including audio/video processing, speech-to-text transcription, chunking, embeddings, vector stores, retrieval, prompt engineering, and LLM-based question answering.