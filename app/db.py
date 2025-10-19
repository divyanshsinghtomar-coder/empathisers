from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import settings


class EmotionLog:
    def __init__(
        self,
        timestamp: str,
        user_message: str,
        emotion: str,
        emergency_level: str,
        user_id: Optional[str] = None,
    ) -> None:
        self.timestamp = timestamp
        self.user_message = user_message
        self.emotion = emotion
        self.emergency_level = emergency_level
        self.user_id = user_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "user_message": self.user_message,
            "emotion": self.emotion,
            "emergency_level": self.emergency_level,
            "user_id": self.user_id,
        }


class Database:
    def __init__(self) -> None:
        self.backend: str
        self._deta_base = None
        self._sqlite_conn: Optional[sqlite3.Connection] = None

        if settings.use_sqlite:
            self.backend = "sqlite"
            self._init_sqlite()
            return

        # Try Deta Base if project key is present
        if settings.deta_project_key:
            try:
                from deta import Deta  # type: ignore

                deta = Deta(settings.deta_project_key)
                self._deta_base = deta.Base(settings.deta_base_name)
                self.backend = "deta"
                return
            except Exception:
                # Fallback to SQLite if Deta init fails
                pass

        self.backend = "sqlite"
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        os.makedirs(os.path.dirname(settings.sqlite_path), exist_ok=True)
        self._sqlite_conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
        self._sqlite_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS emotion_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                user_message TEXT NOT NULL,
                emotion TEXT NOT NULL,
                emergency_level TEXT NOT NULL
            )
            """
        )
        self._sqlite_conn.commit()

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def log_emotion(
        self,
        user_message: str,
        emotion: str,
        emergency_level: str,
        user_id: Optional[str] = None,
    ) -> str:
        timestamp = self._utc_now_iso()

        if self.backend == "deta":
            item = {
                "timestamp": timestamp,
                "user_id": user_id,
                "user_message": user_message,
                "emotion": emotion,
                "emergency_level": emergency_level,
            }
            self._deta_base.put(item)  # type: ignore[attr-defined]
            return timestamp

        assert self._sqlite_conn is not None
        self._sqlite_conn.execute(
            "INSERT INTO emotion_logs (timestamp, user_id, user_message, emotion, emergency_level) VALUES (?, ?, ?, ?, ?)",
            (timestamp, user_id, user_message, emotion, emergency_level),
        )
        self._sqlite_conn.commit()
        return timestamp

    def get_history(self, user_id: Optional[str] = None, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        if self.backend == "deta":
            # Deta Base does not natively support order by, we fetch and sort client-side
            q = None
            if user_id:
                q = {"user_id": user_id}
            res = self._deta_base.fetch(q)  # type: ignore[attr-defined]
            items = res.items
            while res.last:
                res = self._deta_base.fetch(q, last=res.last)  # type: ignore[attr-defined]
                items.extend(res.items)

            items.sort(key=lambda x: x.get("timestamp", ""))
            if last_n is not None:
                items = items[-last_n:]
            # Ensure consistent shape
            return [
                {
                    "timestamp": it.get("timestamp"),
                    "user_message": it.get("user_message"),
                    "emotion": it.get("emotion"),
                    "emergency_level": it.get("emergency_level"),
                    "user_id": it.get("user_id"),
                }
                for it in items
            ]

        assert self._sqlite_conn is not None
        cursor = self._sqlite_conn.cursor()
        if user_id:
            cursor.execute(
                "SELECT timestamp, user_message, emotion, emergency_level, user_id FROM emotion_logs WHERE user_id = ? ORDER BY timestamp ASC",
                (user_id,),
            )
        else:
            cursor.execute(
                "SELECT timestamp, user_message, emotion, emergency_level, user_id FROM emotion_logs ORDER BY timestamp ASC"
            )
        rows = cursor.fetchall()
        if last_n is not None:
            rows = rows[-last_n:]
        return [
            {
                "timestamp": r[0],
                "user_message": r[1],
                "emotion": r[2],
                "emergency_level": r[3],
                "user_id": r[4],
            }
            for r in rows
        ]


db = Database()


