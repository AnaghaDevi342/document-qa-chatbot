import json
from pathlib import Path
import requests
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://127.0.0.1:8000"
QUESTIONS_FILE = Path(__file__).with_name("questions.json")


def load_questions():
    return json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))


def login():
    username = os.getenv("APP_USERNAME")
    password = os.getenv("APP_PASSWORD")

    response = requests.post(
        f"{BASE_URL}/auth/login",
        json={"username": username, "password": password},
        timeout=30,
    )
    response.raise_for_status()

    return response.json()["access_token"]


def headers(token):
    return {"Authorization": f"Bearer {token}"}


def check_known_answer(answer, expected_keywords):
    answer_lower = answer.lower()
    return all(keyword.lower() in answer_lower for keyword in expected_keywords)


def check_unknown_answer(answer):
    answer_lower = answer.lower()

    refusal_phrases = [
        "do not have enough information",
        "don't have enough information",
        "not enough information",
        "cannot answer",
        "can't answer",
        "not available in the uploaded documents",
        "not found in the uploaded documents",
        "do not mention",
        "does not mention",
    ]

    return any(phrase in answer_lower for phrase in refusal_phrases)
