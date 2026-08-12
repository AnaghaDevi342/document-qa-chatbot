from elasticsearch import Elasticsearch


INDEX_NAME = "test_gapblue"


client = Elasticsearch(
    "http://localhost:9200"
)


if not client.ping():
    raise RuntimeError(
        "Could not connect to Elasticsearch"
    )


if client.indices.exists(index=INDEX_NAME):
    client.indices.delete(index=INDEX_NAME)


mapping = {
    "properties": {
        "text": {
            "type": "text"
        },
        "embedding": {
            "type": "dense_vector",
            "dims": 3072,
            "index": True,
            "similarity": "cosine",
        },
    }
}


response = client.indices.create(
    index=INDEX_NAME,
    mappings=mapping,
)


print("Index created successfully")
print(response)