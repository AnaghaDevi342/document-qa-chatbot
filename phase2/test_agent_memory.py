from phase2.app.agent import AgentService

def main():

    agent = AgentService()

    conversation_id = "test-conversation-1"

    print("\n" + "=" * 60)
    print("QUESTION 1")
    print("=" * 60)

    response_1 = agent.invoke(
        "What is the Day 5 project?",
        conversation_id,
    )

    print("\nANSWER 1:")
    print(response_1["answer"])

    print("\n" + "=" * 60)
    print("QUESTION 2")
    print("=" * 60)

    response_2 = agent.invoke(
        "What technologies does it use?",
        conversation_id,
    )

    print("\nANSWER 2:")
    print(response_2["answer"])


if __name__ == "__main__":
    main()