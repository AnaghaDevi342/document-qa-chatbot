import requests

from common import BASE_URL, headers, login

def main():
    token = login()
    auth_headers = headers(token)

    # First question establishes the topic in conversation memory.
    first = requests.post(
        f"{BASE_URL}/chat",
        headers=auth_headers,
        json={
            "question": "What is the Day 5 project?",
        },
        timeout=120,
    )
    first.raise_for_status()

    first_data = first.json()
    conversation_id = first_data["conversation_id"]

    # Follow-up deliberately uses "it" so that Phase 2 memory is tested.
    second = requests.post(
        f"{BASE_URL}/chat",
        headers=auth_headers,
        json={
            "question": "What technologies does it use?",
            "conversation_id": conversation_id,
        },
        timeout=120,
    )
    second.raise_for_status()

    second_data = second.json()
    answer = second_data.get("answer", "").lower()

    expected = [
        "fastapi",
        "postgresql",
        "redis",
        "pandas",
        "docker",
    ]

    passed = all(item in answer for item in expected)

    print("=" * 70)
    print("PHASE 2 CONVERSATIONAL MEMORY TEST")
    print("=" * 70)
    print(f"Conversation ID: {conversation_id}")
    print(f"Answer: {second_data.get('answer', '')}")
    print(f"Passed: {passed}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
