from phase2.app.agent import AgentService

def main():

    agent_service = AgentService()

    response = agent_service.invoke(
        "What is the Day 5 project?"
    )

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(response["output"])


if __name__ == "__main__":
    main()