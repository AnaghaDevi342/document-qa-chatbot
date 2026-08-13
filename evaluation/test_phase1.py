import json
from pathlib import Path

import requests

from common import BASE_URL, headers, load_questions, login, check_known_answer, check_unknown_answer

def main():
    token = login()

    results = []

    for item in load_questions():
        response = requests.post(
            f"{BASE_URL}/chat",
            headers=headers(token),
            json={"question": item["question"]},
            timeout=120,
        )

        result = {
            "id": item["id"],
            "question": item["question"],
            "http_status": response.status_code,
        }

        if response.ok:
            data = response.json()
            answer = data.get("answer", "")

            if item["type"] == "known":
                passed = check_known_answer(
                    answer,
                    item["expected_keywords"],
                )
            else:
                passed = check_unknown_answer(answer)

            result.update({
                "answer": answer,
                "passed": passed,
                "sources": data.get("sources", []),
            })
        else:
            result.update({
                "answer": "",
                "passed": False,
                "error": response.text,
            })

        results.append(result)

        print("=" * 70)
        print(f"Q{item['id']}: {item['question']}")
        print(f"Status : {result['http_status']}")
        print(f"Passed : {result['passed']}")
        print(f"Answer : {result.get('answer', '')}")

    output = Path(__file__).with_name("results") / "phase1_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    passed = sum(item["passed"] for item in results)
    print("\n" + "=" * 70)
    print(f"PHASE 1 EVALUATION: {passed}/{len(results)} PASSED")
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
