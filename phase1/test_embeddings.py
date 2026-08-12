import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=f"{os.getenv('AZURE_OPENAI_ENDPOINT')}/openai/v1/",
)


response = client.embeddings.create(
    model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    input="What is the employee leave policy?",
)


embedding = response.data[0].embedding


print("Embedding generated successfully")
print("Dimensions:", len(embedding))
print("First 5 values:", embedding[:5])