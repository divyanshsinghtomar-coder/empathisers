import os
import sys
import time
from typing import Tuple, Deque, List
from collections import deque
import re

import pyttsx3
import speech_recognition as sr

# Ensure project root on path to import Stage 1 helpers
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.realtime_emotion import analyze_text_emotion, map_to_level, capture_speech_text
from twilio.rest import Client as TwilioClient


def init_tts() -> pyttsx3.Engine:
    engine = pyttsx3.init()
    engine.setProperty('rate', 155)
    engine.setProperty('volume', 0.9)
    return engine


def speak(engine: pyttsx3.Engine, text: str) -> None:
    engine.say(text)
    engine.runAndWait()


def send_emergency_sms(emotion: str, voice_text: str) -> None:
    """Send an emergency SMS via Twilio when High level is detected.

    The credentials below are placeholders. For production, load from env or a secure store.
    """
    ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "YOUR_TWILIO_SID")
    AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "YOUR_TWILIO_AUTH_TOKEN")
    FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "+1XXXXXXXXXX")
    TO_NUMBER = os.getenv("TWILIO_TO_NUMBER", "+91XXXXXXXXXX")

    message_body = (
        f"🚨 AI Guardian Alert: The user seems in distress. Emotion: {emotion}. "
        f"Message: {voice_text}. Please check immediately."
    )

    client = TwilioClient(ACCOUNT_SID, AUTH_TOKEN)
    client.messages.create(body=message_body, from_=FROM_NUMBER, to=TO_NUMBER)

def get_stage1_output(recognizer: sr.Recognizer, mic: sr.Microphone) -> Tuple[str, str, str]:
    voice_text = capture_speech_text(recognizer, mic)
    emotion = analyze_text_emotion(voice_text)
    level = map_to_level(emotion)
    return voice_text, emotion, level


def build_prompt(voice_text: str, emotion: str, level: str, recent_replies: List[str], nonce: str) -> str:
    # Compose the dynamic prompt including latest input and a soft constraint to avoid repeating past replies
    recent_block = "\n".join(f"- {r}" for r in recent_replies[-5:]) if recent_replies else "(none)"
    return f"""
You are 'AI Guardian 2.0', a caring, empathetic, human-like companion. 
Respond in natural, friendly speech suitable for TTS.

Guidelines:
- Tailor responses based on user's emotion and emergency level.
- Responses must be dynamic, varied, and context-aware; do NOT repeat the same sentences.
- Avoid generic phrases like "Take a deep breath" repeatedly.
- Keep responses short, calm, human-like.
- Adapt tone based on emergency level:
   * Low: friendly, motivational, uplifting.
   * Medium: empathetic, calming, suggest coping strategies.
   * High: firm but caring, provide safety instructions.
- Include deep breathing or grounding techniques creatively when appropriate.
- Do not mention AI or include code/markdown in your reply.

Context:
- User Emotion: {emotion}
- Emergency Level: {level}
- User Said: "{voice_text}"

Recent replies (avoid repeating wording):
{recent_block}

Task: Generate a short, empathetic reply suitable for TTS.
Nonce: {nonce}
""".strip()


def generate_gpt_reply(voice_text: str, emotion: str, level: str, recent_replies: List[str]) -> str:
    # Use OpenAI SDK (env var OPENAI_API_KEY required)
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Please set the key to enable dynamic GPT responses.")

    client = OpenAI(api_key=api_key)
    nonce = str(time.time())
    prompt = build_prompt(voice_text, emotion, level, recent_replies, nonce)

    # Use responses API (Chat Completions style with newer SDK)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an empathetic companion and respond concisely for TTS."},
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,
        top_p=0.95,
        presence_penalty=0.6,
        frequency_penalty=0.6,
        max_tokens=90,
    )
    text = resp.choices[0].message.content.strip()

    # One-time anti-repetition check: if too similar to recent replies, try once more
    def _norm(s: str) -> List[str]:
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        return [t for t in s.split() if t]

    def _similar(a: str, b: str) -> float:
        ta, tb = set(_norm(a)), set(_norm(b))
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        uni = len(ta | tb)
        return inter / uni

    if any(_similar(text, r) >= 0.7 for r in recent_replies[-5:]):
        # regenerate with a new nonce
        nonce = str(time.time())
        prompt = build_prompt(voice_text, emotion, level, recent_replies, nonce)
        resp2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an empathetic companion and respond concisely for TTS."},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
            top_p=0.95,
            presence_penalty=0.6,
            frequency_penalty=0.6,
            max_tokens=90,
        )
        text = resp2.choices[0].message.content.strip()

    return text


def run_loop() -> None:
    # Load .env (optional)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    engine = init_tts()
    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone()
    except Exception:
        print("Microphone not available.")
        return

    print("AI Voice Chatbot (GPT TTS): speaking-only responses. Ctrl+C to stop.")
    # Keep a rolling list of recent replies (medium history) to discourage repetition
    recent_replies: Deque[str] = deque(maxlen=8)
    while True:
        try:
            # Pause to avoid immediate re-trigger
            time.sleep(0.15)
            voice_text, emotion, level = get_stage1_output(recognizer, mic)
            # If no speech captured, prompt again without calling GPT
            if not voice_text:
                continue
            # Generate GPT reply
            reply = generate_gpt_reply(voice_text, emotion, level, list(recent_replies))
            # Speak reply
            speak(engine, reply)
            # Update history
            recent_replies.append(reply)

            # Send SMS if High emergency
            if level.lower() == "high":
                try:
                    send_emergency_sms(emotion, voice_text)
                    print("🚨 Emergency SMS sent to guardian!")
                except Exception:
                    # Continue loop even if SMS sending fails
                    print("Warning: Failed to send emergency SMS. Continuing.")
        except KeyboardInterrupt:
            print("\nStopping.")
            break
        except RuntimeError as e:
            # Missing API key or critical config issue
            print(str(e))
            time.sleep(1.0)
            continue
        except Exception:
            # Swallow transient errors and continue
            time.sleep(0.4)
            continue


if __name__ == "__main__":
    run_loop()


