import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Deque, Tuple

import cv2
_DEEPFACE_IMPORT_ERROR = None
try:
    from deepface import DeepFace  # type: ignore
    _HAS_DEEPFACE = True
except Exception as _e:
    DeepFace = None  # type: ignore
    _HAS_DEEPFACE = False
    _DEEPFACE_IMPORT_ERROR = str(_e)
import speech_recognition as sr
from colorama import Fore, Style, init as colorama_init
from textblob import TextBlob
import re
from collections import deque, Counter


mp_face_mesh = None  # Mediapipe disabled; using DeepFace instead


@dataclass
class DetectionResult:
    emotion: str
    level: str
    text: str
    timestamp: str


def classify_expression_from_landmarks(landmarks, image_width: int, image_height: int) -> str:
    # Simple heuristic using mouth aspect ratio and eyebrow distance as proxies
    # This is a lightweight placeholder; for production use, a trained model is recommended.
    if not landmarks:
        return "neutral"

    # Indices for mediapipe face mesh (approximate regions)
    mouth_upper_idx = 13
    mouth_lower_idx = 14
    left_brow_idx = 52
    right_brow_idx = 282
    nose_tip_idx = 1

    def to_xy(idx: int):
        lm = landmarks[idx]
        return lm.x * image_width, lm.y * image_height

    mx1, my1 = to_xy(mouth_upper_idx)
    mx2, my2 = to_xy(mouth_lower_idx)
    mouth_gap = abs(my2 - my1)

    bx1, by1 = to_xy(left_brow_idx)
    bx2, by2 = to_xy(nose_tip_idx)
    brow_to_nose_left = abs(by2 - by1)

    rx1, ry1 = to_xy(right_brow_idx)
    brow_to_nose_right = abs(by2 - ry1)

    brow_avg = (brow_to_nose_left + brow_to_nose_right) / 2.0

    # Heuristics thresholds tuned loosely; adjust as needed
    if mouth_gap > 12 and brow_avg > 55:
        return "fearful"  # open mouth + raised brows
    if mouth_gap > 10:
        return "happy"  # smile proxy
    if brow_avg < 38:
        return "angry"  # lowered brows
    return "neutral"


def _normalize_text(raw: str) -> str:
    t = raw.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def analyze_text_emotion(text: str) -> str:
    t = _normalize_text(text)
    if not t:
        return "neutral"

    # Crisis keywords → High via "panic"
    crisis_keywords = {
        "suicide",
        "sucide",
        "suicidal",
        "commit suicide",
        "take my life",
        "kill myself",
        "killing myself",
        "end my life",
        "want to die",
        "i should die",
        "i want to die",
        "end it all",
        "self-harm",
        "self harm",
        "cut myself",
        "hurt myself",
        "hopeless",
        "no reason to live",
    }
    if any(k in t for k in crisis_keywords):
        return "panic"

    # Extensive keyword categories
    KEYWORDS_HAPPY_LOW = {
        "happy", "great", "good", "amazing", "fine", "awesome", "excellent", "wonderful", "fun",
        "joyful", "cool", "glad", "satisfied", "cheerful", "delighted", "thrilled", "blessed",
        "lucky", "content", "relaxed", "excited", "energetic", "upbeat", "optimistic", "proud",
        "peaceful", "carefree", "confident", "pleased", "radiant", "lighthearted", "playful",
        "encouraged", "motivated",
    }
    KEYWORDS_NEUTRAL_LOW = {
        "okay", "normal", "alright", "nothing", "so-so", "meh", "neutral", "bored", "average",
        "usual", "standard", "indifferent", "steady", "balanced", "uneventful", "calm", "passive",
        "mellow", "typical", "regular", "routine",
    }
    KEYWORDS_SAD_MEDIUM = {
        "sad", "lonely", "down", "unhappy", "miserable", "depressed", "hopeless", "low", "upset",
        "tearful", "anxious", "tired", "gloomy", "hurt", "sorrow", "distressed", "disappointed",
        "rejected", "exhausted", "frustrated", "regret", "discouraged", "melancholic", "helpless",
        "grieving", "heartbroken", "overwhelmed", "isolated", "ashamed",
    }
    KEYWORDS_ANGRY_HIGH = {
        "angry", "mad", "frustrated", "annoyed", "irritated", "furious", "hate", "rage", "bothered",
        "resentful", "infuriated", "vexed", "offended", "hostile", "bitter", "enraged", "exasperated",
        "outraged", "tense", "irate", "aggressive", "cross", "mad at", "provoked", "insulted",
        "livid", "fuming",
    }
    KEYWORDS_FEAR_HIGH = {
        "scared", "afraid", "panic", "terrified", "nervous", "anxious", "stressed", "worried",
        "danger", "threat", "unsafe", "hurt", "die", "help", "frightened", "uneasy", "alarmed",
        "fearful", "overwhelmed", "tremble", "shaking", "insecure", "threatened", "intimidated",
        "horror", "startled", "shocked", "vulnerable", "concerned", "jittery",
    }
    KEYWORDS_CONFUSED_MEDIUM = {
        "confused", "don't know", "unsure", "lost", "dilemma", "what should i do", "can't decide",
        "perplexed", "puzzled", "uncertain", "hesitant", "undecided", "baffled", "bewildered",
        "unsure how", "doubtful", "tricky", "complicated", "uncertain situation", "unclear",
        "indecisive", "lost track", "unsure feeling", "conflicted",
    }
    KEYWORDS_EXCITED_LOW = {
        "excited", "pumped", "thrilled", "energetic", "motivated", "eager", "can't wait", "delighted",
        "enthusiastic", "inspired", "ready", "charged", "adventurous", "spirited", "elated",
        "exhilarated", "encouraged", "active", "lively", "dynamic", "vigorous", "ecstatic",
    }

    if any(k in t for k in KEYWORDS_FEAR_HIGH):
        return "fearful"
    if any(k in t for k in KEYWORDS_ANGRY_HIGH):
        return "angry"
    if any(k in t for k in KEYWORDS_SAD_MEDIUM):
        return "sad"
    if any(k in t for k in KEYWORDS_CONFUSED_MEDIUM):
        return "confused"
    if any(k in t for k in KEYWORDS_EXCITED_LOW) or any(k in t for k in KEYWORDS_HAPPY_LOW):
        return "happy"
    if any(k in t for k in KEYWORDS_NEUTRAL_LOW):
        return "neutral"

    # Fallback to sentiment
    polarity = float(TextBlob(t).sentiment.polarity)
    if polarity > 0.2:
        return "happy"
    if polarity < -0.2:
        return "sad"
    return "neutral"


def combine_emotions(face_emotion: str, text_emotion: str) -> str:
    # Normalize DeepFace labels
    face_norm = (face_emotion or "").lower()
    if face_norm == "fear":
        face_norm = "fearful"
    if face_norm == "surprise":
        # treat strong surprise as fear proxy
        face_norm = "fearful"

    # Prioritize High severity
    if face_norm in {"fearful", "angry"}:
        return face_norm
    if text_emotion in {"panic", "fearful", "angry"}:
        return "fearful" if text_emotion != "angry" else "angry"

    # Medium next
    if text_emotion in {"sad", "confused", "stressed", "lonely"} or face_norm in {"sad"}:
        return "sad" if text_emotion == "sad" or face_norm == "sad" else text_emotion

    # Low otherwise
    if text_emotion in {"happy", "neutral"} or face_norm in {"happy", "neutral"}:
        return "happy" if (text_emotion == "happy" or face_norm == "happy") else "neutral"
    return "neutral"


def map_to_level(emotion: str) -> str:
    # Mapping per spec:
    # happy, excited, neutral -> Low
    # sad, confused -> Medium
    # fear, angry -> High
    e = emotion.lower()
    if e in {"fearful", "panic", "angry"}:
        return "High"
    if e in {"sad", "confused", "stressed", "lonely"}:
        return "Medium"
    return "Low"


def capture_speech_text(recognizer: sr.Recognizer, mic: sr.Microphone) -> str:
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, phrase_time_limit=6)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""


def color_for_level(level: str) -> str:
    l = level.lower()
    if l == "high":
        return Fore.RED
    if l == "medium":
        return Fore.YELLOW
    return Fore.GREEN


def run_loop() -> None:
    colorama_init(autoreset=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Webcam not available. Facial expression disabled.")
        cap = None

    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone()
    except Exception:
        print("Warning: Microphone not available. Voice input disabled.")
        mic = None  # type: ignore

    print("\nAI Guardian 2.0 – Real-time Emotion (Press Enter to capture, 'exit' to quit)\n")
    if not _HAS_DEEPFACE:
        note = "Note: DeepFace not available. Using audio-only detection."
        if _DEEPFACE_IMPORT_ERROR:
            note += f" Import error: {_DEEPFACE_IMPORT_ERROR}"
        print(note)

    face_mesh_ctx = None
    # Sliding window for smoothing final emotions
    recent_finals: Deque[str] = deque(maxlen=3)
    try:
        while True:
            cmd = input("Press Enter to capture (exit to quit): ").strip().lower()
            if cmd == "exit":
                break

            frame = None
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    print("Warning: Failed to read frame; facial expression disabled this round.")
                    frame = None

            face_emotion = "neutral"
            face_fear_score: float = 0.0
            if frame is not None and _HAS_DEEPFACE:
                try:
                    # DeepFace returns dict with dominant_emotion
                    df_res = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
                    # DeepFace can return list or dict depending on version
                    if isinstance(df_res, list) and df_res:
                        face_emotion = str(df_res[0].get("dominant_emotion", "neutral")).lower()
                        emotions_dict = df_res[0].get("emotion") or {}
                    elif isinstance(df_res, dict):
                        face_emotion = str(df_res.get("dominant_emotion", "neutral")).lower()
                        emotions_dict = df_res.get("emotion") or {}
                    else:
                        emotions_dict = {}

                    # Extract fear probability for thresholding (0..100)
                    if isinstance(emotions_dict, dict):
                        fear_val = emotions_dict.get("fear")
                        try:
                            face_fear_score = float(fear_val) / 100.0 if fear_val is not None else 0.0
                        except Exception:
                            face_fear_score = 0.0
                except Exception:
                    face_emotion = "neutral"
                    face_fear_score = 0.0

            text = ""
            if mic is not None:
                try:
                    text = capture_speech_text(recognizer, mic)
                except Exception:
                    text = ""

            text_emotion = analyze_text_emotion(text)

            # Remap DeepFace labels with confidence thresholds
            def remap_deepface(label: str, fear_score: float) -> str:
                l = (label or "").lower()
                if l in {"happy"}:
                    return "happy"
                if l in {"neutral"}:
                    return "neutral"
                if l in {"sad"}:
                    return "sad"
                if l in {"angry"}:
                    return "angry"
                if l in {"fear"}:
                    # treat as anxiety unless strong confidence
                    return "fearful" if fear_score >= 0.7 else "stressed"
                if l in {"surprise", "disgust"}:
                    return "neutral"
                return l or "neutral"

            remapped_face = remap_deepface(face_emotion, face_fear_score)

            # Combine per rules: prioritize voice for Medium if face shows fear/anxiety
            # High only if both facial extreme fear and voice indicates fear/panic
            face_extreme_fear = (remapped_face == "fearful" and face_fear_score >= 0.7)

            def combine(face_label: str, voice_label: str) -> str:
                # Crisis in voice always wins → fearful
                if voice_label in {"panic", "fearful"}:
                    return "fearful"

                # Face angry → angry (High)
                if face_label == "angry":
                    return "angry"

                # Face fear/anxiety but voice sad/anxious → prioritize voice Medium
                if face_label in {"fearful", "stressed"} and voice_label in {"sad", "confused", "stressed", "lonely"}:
                    return voice_label

                # High fear only if extreme fear on face and voice also fear/panic
                if face_extreme_fear and voice_label in {"panic", "fearful"}:
                    return "fearful"

                # If either is sad-like, return sad
                if voice_label in {"sad", "confused", "stressed", "lonely"} or face_label == "sad":
                    return "sad"

                # Favor happiness if any is happy
                if voice_label == "happy" or face_label == "happy":
                    return "happy"

                # Default neutral
                return "neutral"

            final_emotion = combine(remapped_face, text_emotion)

            # Smoothing over last 3 final emotions
            recent_finals.append(final_emotion)
            if len(recent_finals) >= 2:
                most_common: Tuple[str, int] = Counter(recent_finals).most_common(1)[0]
                final_emotion = most_common[0]

            level = map_to_level(final_emotion)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            c = color_for_level(level)
            print(f"[{ts}] Face={face_emotion}, Voice='{text or '-'}' -> Detected emotion: {final_emotion}")
            print(f"Emergency level: {c}{level}{Style.RESET_ALL}\n")
    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        run_loop()
    except KeyboardInterrupt:
        print("\nExiting.")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        sys.exit(0)


