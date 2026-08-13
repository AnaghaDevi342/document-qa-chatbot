from langchain_openai import AzureChatOpenAI

from phase2.app.config import settings


model = AzureChatOpenAI(
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_api_key,
    azure_deployment=(
        settings.azure_openai_chat_deployment
    ),
    api_version=settings.azure_openai_version,
    temperature=0,
)

response = model.invoke(
    "Say hello in one sentence."
)

print(response.content)