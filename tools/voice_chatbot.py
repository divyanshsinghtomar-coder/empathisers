import time
from typing import Tuple
import os
import sys

import pyttsx3

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.realtime_emotion import analyze_text_emotion, map_to_level, capture_speech_text
import speech_recognition as sr


def generate_response(emotion: str, level: str) -> str:
    e = (emotion or "").lower()
    l = (level or "").capitalize()
    if l == "High":
        return "I'm here with you. Stay calm and breathe slowly. If you're in danger, seek immediate help or call local emergency services."
    if l == "Medium":
        if e in {"sad", "confused", "stressed", "lonely"}:
            return "I hear you. Let's take a few deep breaths together. You are not alone. Would reaching out to a trusted person help right now?"
        return "I can sense discomfort. We can slow down and breathe. I'm here with you."
    # Low
    return "I hear you. Take a deep breath. You're doing great. What would help you feel a bit better right now?"


def speak(engine: pyttsx3.Engine, text: str) -> None:
    engine.say(text)
    engine.runAndWait()


def get_stage1_output(recognizer: sr.Recognizer, mic: sr.Microphone) -> Tuple[str, str, str]:
    text = capture_speech_text(recognizer, mic)
    emotion = analyze_text_emotion(text)
    level = map_to_level(emotion)
    return text, emotion, level


def run_loop() -> None:
    engine = pyttsx3.init()
    engine.setProperty('rate', 155)
    engine.setProperty('volume', 0.9)

    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone()
    except Exception:
        print("Microphone not available.")
        return

    print("AI Voice Chatbot (Stage 2): speaking only responses. Press Ctrl+C to stop.")
    while True:
        try:
            print("Listening...")
            voice_text, emotion, level = get_stage1_output(recognizer, mic)
            print(f"Heard: {voice_text or '-'} | Emotion={emotion} Level={level}")
            response = generate_response(emotion, level)
            speak(engine, response)
            time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nStopping.")
            break
        except Exception:
            # continue loop on any transient error
            time.sleep(0.5)
            continue


if __name__ == "__main__":
    run_loop()


