from __future__ import annotations

from textblob import TextBlob


STRESS_KEYWORDS = {
    "stress",
    "stressed",
    "anxious",
    "anxiety",
    "overwhelmed",
    "panic",
    "panicking",
    "panic attack",
    "burned out",
    "burnout",
}


def analyze_sentiment(user_message: str) -> dict:
    """Analyze sentiment and map to emotion + emergency level.

    Rules (prototype):
    - Polarity > 0.2 => happy
    - Polarity < -0.2 => sad
    - Else => neutral
    Emergency mapping:
    - sad and polarity < -0.6 => Medium
    - neutral => Low
    - happy => Low
    - If stress keywords detected => emotion "stressed", emergency "Medium"
    """
    polarity = float(TextBlob(user_message).sentiment.polarity)

    # Heuristic override for stress signals
    lower_msg = user_message.lower()
    if any(keyword in lower_msg for keyword in STRESS_KEYWORDS):
        return {"emotion": "stressed", "emergency_level": "Medium", "polarity": polarity}

    if polarity > 0.2:
        emotion = "happy"
        emergency_level = "Low"
    elif polarity < -0.2:
        emotion = "sad"
        emergency_level = "Medium" if polarity < -0.6 else "Low"
    else:
        emotion = "neutral"
        emergency_level = "Low"

    return {"emotion": emotion, "emergency_level": emergency_level, "polarity": polarity}


