import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")


if not api_key:
    raise ValueError("AZURE_OPENAI_API_KEY is not configured")

if not endpoint:
    raise ValueError("AZURE_OPENAI_ENDPOINT is not configured")

if not deployment:
    raise ValueError(
        "AZURE_OPENAI_CHAT_DEPLOYMENT is not configured"
    )


client = OpenAI(
    api_key=api_key,
    base_url=f"{endpoint}/openai/v1/"
)


response = client.chat.completions.create(
    model=deployment,
    messages=[
        {
            "role": "user",
            "content": "Say hello in one sentence."
        }
    ]
)


print(response.choices[0].message.content)