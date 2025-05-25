# ------------------------------ IMPORTS ------------------------------
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
import io
import numpy as np
import joblib
import pickle
import torch
import cv2
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model as keras_load_model
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm  # Progress bar

# ------------------------------ LOAD MODELS ------------------------------
text_model = joblib.load("text.h5")
audio_model = keras_load_model("depression_detection_model.h5")
video_model = keras_load_model("my_model.h5")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to("cpu")

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

# ------------------------------ VIDEO TO AUDIO ------------------------------
def convert_video_to_audio(video_path, audio_path):
    print("\n🔄 Converting video to audio...")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, verbose=False)
    print("✅ Audio extraction complete!")

# ------------------------------ AUDIO TO TEXT ------------------------------
def convert_audio_to_wav(input_audio_path):
    audio = AudioSegment.from_file(input_audio_path)
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io

def convert_speech_to_text(audio_data):
    print("\n🗣️ Transcribing audio to text...")
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_data) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        print("✅ Transcription complete!")
        return text
    except sr.UnknownValueError:
        print("⚠️ Speech Recognition could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"⚠️ Could not request results; {e}")
        return None

def save_text_to_file(text, file_path):
    with open(file_path, 'w') as file:
        file.write(text)

# ------------------------------ VIDEO ANALYSIS ------------------------------
def analyze_video(video_path, frame_skip=10):
    print("\n🎥 Analyzing video for emotions...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    emotion_data = {emotion: [] for emotion in label_encoder.classes_}
    frame_emotion_series = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    frame_count = 0
    with tqdm(total=total_frames, desc="📷 Processing frames", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        face = cv2.resize(gray[y:y+h, x:x+w], (48, 48)).astype('float32') / 255.0
                        face = face.reshape(1, 48, 48, 1)
                        preds = video_model.predict(face, verbose=0)[0]
                        emotion_dict = {label_encoder.classes_[i]: preds[i] * 100 for i in range(len(preds))}
                        frame_emotion_series.append(emotion_dict)
                        for emotion, value in emotion_dict.items():
                            emotion_data[emotion].append(value)
                        break
            frame_count += 1
            pbar.update(1)

    cap.release()
    avg_emotions = {emo: np.mean(vals) if vals else 0 for emo, vals in emotion_data.items()}
    depression_video_score = avg_emotions.get("sad", 0)
    return depression_video_score, avg_emotions, frame_emotion_series

# ------------------------------ TEXT ANALYSIS ------------------------------
def analyze_text(text_input):
    print("\n🧾 Analyzing text for suicide risk...")
    with torch.no_grad():
        encoded = tokenizer([text_input], padding=True, truncation=True, return_tensors='pt', max_length=512)
        features = bert_model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()
    suicide_prob = text_model.predict_proba(features)[0][1] * 100
    print(f"✅ Suicide risk score: {suicide_prob:.2f}%")
    return suicide_prob

# ------------------------------ AUDIO ANALYSIS ------------------------------
def analyze_audio(audio_path):
    print("\n🎧 Analyzing audio for depression...")
    y, sr = librosa.load(audio_path, sr=None)
    expected_n_mfcc = audio_model.input_shape[-1]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=expected_n_mfcc)

    if mfcc.shape[1] < 40000:
        pad = np.zeros((mfcc.shape[0], 40000 - mfcc.shape[1]))
        mfcc = np.concatenate((mfcc, pad), axis=1)
    else:
        mfcc = mfcc[:, :40000]

    mfcc = mfcc.T.reshape(1, 40000, expected_n_mfcc)
    depression_prob = audio_model.predict(mfcc, verbose=0)[0][0] * 100
    print(f"✅ Depression from audio: {depression_prob:.2f}%")
    return depression_prob

# ------------------------------ COMBINED ASSESSMENT ------------------------------
def combined_assessment(video_path, text_input, audio_path):
    video_score, video_emotions, frame_emotion_series = analyze_video(video_path)
    text_score = analyze_text(text_input)
    audio_score = analyze_audio(audio_path)

    final_score = (video_score * 0.4 + text_score * 0.3 + audio_score * 0.3)
    print(f"\n🧠 Final Mental Health Risk Score: {final_score:.2f}%")

    return {
        "video_score": video_score,
        "text_score": text_score,
        "audio_score": audio_score,
        "final_score": final_score,
        "emotion_breakdown": video_emotions,
        "frame_emotion_series": frame_emotion_series
    }

# ------------------------------ PLOTTING ------------------------------
def show_plots(results):
    df_emotions = pd.DataFrame(results['frame_emotion_series'])
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))

    # Define unique colors for each emotion (customize as needed)
    emotion_color_map = {
        'Angry': 'red',
        'Disgust': 'green',
        'Fear': 'purple',
        'Happy': 'gold',
        'Neutral': 'gray',
        'Sad': 'blue',
        'Surprise': 'orange'
    }

    # Extract emotion names and values
    emotion_names = list(results['emotion_breakdown'].keys())
    emotion_values = list(results['emotion_breakdown'].values())
    bar_colors = [emotion_color_map.get(emo, 'black') for emo in emotion_names]  # fallback color is black

    # Plotting
    axs[0, 0].bar(emotion_names, emotion_values, color=bar_colors, edgecolor='black')
    axs[0, 0].set_title("Emotion Distribution from Video", fontsize=14, fontweight='bold')
    axs[0, 0].set_ylabel("Probability (%)", fontsize=12)
    axs[0, 0].tick_params(axis='x', rotation=45)
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels
    for i, value in enumerate(emotion_values):
        axs[0, 0].text(i, value + 1, f"{value:.1f}%", ha='center', va='bottom', fontsize=10)


    # Risk Scores
    labels = ['Facial', 'Text', 'Tone', 'Average']
    values = [results['video_score'], results['text_score'], results['audio_score'],
              (results['video_score'] + results['text_score'] + results['audio_score']) / 3]
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'orange']

    bars = axs[0, 1].bar(labels, values, color=colors)
    axs[0, 1].set_ylim(0, 100)
    axs[0, 1].set_ylabel("Risk Percentage")
    axs[0, 1].set_title("Multi-modal Suicide/Depression Risk")
    for bar in bars:
        yval = bar.get_height()
        axs[0, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', ha='center')
    axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # Emotion Trends Over Time
    for emotion in df_emotions.columns:
        axs[1, 0].plot(df_emotions[emotion], label=emotion)
    axs[1, 0].set_title("Emotion Trends Over Time")
    axs[1, 0].set_xlabel("Frame Index")
    axs[1, 0].set_ylabel("Confidence (%)")
    axs[1, 0].legend(loc='upper right')
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)

    # Sad Emotion Trend
    if 'Sad' in df_emotions.columns:
        axs[1, 1].plot(df_emotions['Sad'], color='blue', linewidth=2)
        axs[1, 1].set_title("Sad Emotion Trend Over Time")
        axs[1, 1].set_xlabel("Frame Index")
        axs[1, 1].set_ylabel("Sadness Confidence (%)")
        axs[1, 1].grid(True, linestyle='--', alpha=0.5)
    else:
        axs[1, 1].text(0.5, 0.5, "⚠️ 'Sad' emotion not found", ha='center', va='center')
        axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

# ------------------------------ MAIN ------------------------------
def main():
    vfile = "video.mp4"
    audio_path = "audio.mp3"
    text_path = "output.txt"

    convert_video_to_audio(vfile, audio_path)

    audio_data = convert_audio_to_wav(audio_path)
    text = convert_speech_to_text(audio_data)
    if not text:
        print("❌ Cannot proceed without transcribed text.")
        return

    save_text_to_file(text, text_path)
    with open(text_path, 'r') as file:
        content = file.read()

    results = combined_assessment(vfile, content, audio_path)
    show_plots(results)

if __name__ == "__main__":
    main()
