import requests

from common import BASE_URL

def main():
    print("=" * 70)
    print("AUTHENTICATION EVALUATION")
    print("=" * 70)

    response = requests.post(
        f"{BASE_URL}/auth/login",
        json={
            "username": "wrong_user",
            "password": "wrong_password",
        },
        timeout=30,
    )

    print(f"Invalid login status: {response.status_code}")

    if response.status_code != 401:
        raise SystemExit("Expected HTTP 401 for invalid credentials.")

    print("Authentication test passed.")


if __name__ == "__main__":
    main()
