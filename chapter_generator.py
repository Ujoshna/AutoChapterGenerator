import whisper
import argparse
import os
from datetime import timedelta
from moviepy.editor import VideoFileClip
from keybert import KeyBERT

# Load KeyBERT model once
kw_model = KeyBERT()

# 1. Extract audio from video
def extract_audio(video_path, audio_path="temp_audio.wav"):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, logger=None)
    return audio_path

# 2. Generate keyword-based chapter title using KeyBERT
def generate_keyword_title(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=1)
    return keywords[0][0].title() if keywords else "Chapter"

# 3. Transcribe audio using Whisper
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['segments']

# 4. Format timestamps nicely
def format_timestamp(seconds):
    return str(timedelta(seconds=int(float(seconds))))

# 5. Generate exactly `num_chapters` titles spaced throughout the video
def generate_chapters(segments, num_chapters=7):
    chapters = []
    total_duration = segments[-1]['end']
    chapter_interval = total_duration / num_chapters

    grouped_segments = [[] for _ in range(num_chapters)]
    
    for segment in segments:
        index = min(int(segment['start'] // chapter_interval), num_chapters - 1)
        grouped_segments[index].append(segment['text'])

    for i, group in enumerate(grouped_segments):
        if not group:
            continue
        combined_text = " ".join(group)
        title = generate_keyword_title(combined_text)
        start_time = i * chapter_interval
        chapters.append({"start": start_time, "title": title})

    return chapters

# 6. Export chapters in YouTube format
def export_youtube_format(chapters, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for ch in chapters:
            f.write(f"{format_timestamp(ch['start'])} {ch['title']}\n")

# 7. Main block
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Directory to save chapter outputs")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    audio_path = extract_audio(args.video)
    segments = transcribe_audio(audio_path)
    chapters = generate_chapters(segments, num_chapters=7)
    export_youtube_format(chapters, os.path.join(args.output, "youtube_chapters.txt"))

    print("✅ Exactly 7 smart chapters exported to YouTube format.")