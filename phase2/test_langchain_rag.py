from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore

from phase2.app.config import settings


PDF_PATH = "uploads/Week2_Detailed_Guide 2.pdf"

INDEX_NAME = "phase2_documents"


def main():

    loader = PyPDFLoader(PDF_PATH)

    documents = loader.load()

    print(
        f"Loaded {len(documents)} pages"
    )

    # --------------------------------------------------
    # 2. Split documents
    # --------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks"
    )

    # --------------------------------------------------
    # 3. Add metadata
    # --------------------------------------------------

    for chunk in chunks:

        chunk.metadata["source"] = (
            chunk.metadata.get(
                "source",
                PDF_PATH,
            )
        )

        chunk.metadata["document_type"] = "pdf"

    # --------------------------------------------------
    # 4. Azure embeddings
    # --------------------------------------------------

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=(
            settings.azure_openai_endpoint
        ),
        api_key=(
            settings.azure_openai_api_key
        ),
        azure_deployment=(
            settings.azure_openai_embedding_deployment
        ),
        api_version=(
            settings.azure_openai_version
        ),
    )

    # Test embedding first.
    vector = embeddings.embed_query(
        "What is the Day 5 project?"
    )

    print(
        f"Embedding dimensions: {len(vector)}"
    )

    # --------------------------------------------------
    # 5. ElasticsearchStore
    # --------------------------------------------------

    vector_store = (
        ElasticsearchStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=INDEX_NAME,
            es_url=settings.elasticsearch_url,
        )
    )

    print(
        "Documents indexed successfully"
    )

    # --------------------------------------------------
    # 6. Retriever
    # --------------------------------------------------

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 5,
        }
    )

    results = retriever.invoke(
        "What is the Day 5 project?"
    )

    print(
        f"Retrieved {len(results)} documents"
    )

    # --------------------------------------------------
    # 7. Display results
    # --------------------------------------------------

    for index, document in enumerate(
        results,
        start=1,
    ):

        print("\n" + "=" * 60)

        print(
            f"RESULT {index}"
        )

        print(
            "Metadata:",
            document.metadata,
        )

        print(
            "\nContent:"
        )

        print(
            document.page_content[:500]
        )


if __name__ == "__main__":
    main()