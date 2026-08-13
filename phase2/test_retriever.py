"""
from phase2.app.services import VectorStoreService

def main():

    vector_store = VectorStoreService()

    retriever = vector_store.get_retriever()

    results = retriever.invoke(
        "What is the Day 5 project?"
    )

    print(
        f"\nRetrieved {len(results)} documents\n"
    )

    for index, document in enumerate(
        results,
        start=1,
    ):

        print("=" * 60)

        print(f"RESULT {index}")

        print("Metadata:")
        print(document.metadata)

        print("\nContent:")
        print(document.page_content[:500])


if __name__ == "__main__":
    main()
"""

from phase2.app.services import VectorStoreService
from phase2.app.constants import INDEX_NAME, TOP_K


def main():

    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)

    print("INDEX_NAME:", INDEX_NAME)
    print("TOP_K:", TOP_K)

    vector_store = VectorStoreService()

    print("\nVector store created successfully.")

    print("\nTesting similarity_search...")

    results = vector_store.vector_store.similarity_search(
        "What is the Day 5 project?",
        k=5,
    )

    print(
        f"\nSimilarity search returned: {len(results)} documents"
    )

    for index, document in enumerate(
        results,
        start=1,
    ):
        print("\n" + "=" * 60)
        print(f"RESULT {index}")
        print("Metadata:", document.metadata)
        print("Content:")
        print(document.page_content[:500])

    print("\n" + "=" * 60)
    print("Testing retriever...")
    print("=" * 60)

    retriever = vector_store.get_retriever()

    results = retriever.invoke(
        "What is the Day 5 project?"
    )

    print(
        f"\nRetriever returned: {len(results)} documents"
    )


if __name__ == "__main__":
    main()