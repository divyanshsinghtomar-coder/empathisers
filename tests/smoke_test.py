from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def pretty(label: str, resp):
    print(label + ":", resp.status_code, resp.json())


def run():
    r1 = client.post("/detect_emotion", json={"user_message": "I feel lonely today"})
    pretty("/detect_emotion", r1)

    payload = {
        "user_message": "I feel lonely today",
        "emotion": r1.json()["emotion"],
        "emergency_level": r1.json()["emergency_level"],
        "user_id": "tester",
    }
    r2 = client.post("/log_emotion", json=payload)
    pretty("/log_emotion", r2)

    r3 = client.get("/history", params={"user_id": "tester", "last_n": 10})
    pretty("/history", r3)


if __name__ == "__main__":
    run()


