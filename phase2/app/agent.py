from langchain_classic.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)

from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from pathlib import Path
from .services import (
    ChatModelService,
    ToolService,
    vector_store_service,
)
from .constants import TOP_K


SYSTEM_PROMPT = """
You are a document assistant.

Answer questions ONLY using information
from the uploaded documents.

Rules:

1. Use the search_documents tool when the
   question requires document information.

2. Do not invent information.

3. If the documents do not contain the answer,
   clearly say that you do not have enough
   information.

4. Cite the document name and page number.

5. Use the conversation history to understand
   follow-up questions and references such as
   "it", "they", "that project", etc.

6. Keep answers concise and factual.
"""


class AgentService:

    def __init__(self):

        self.llm = ChatModelService().model

        self.tools = [
            ToolService.create_search_tool()
        ]

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_PROMPT,
                ),
                MessagesPlaceholder(
                    variable_name="history"
                ),
                (
                    "human",
                    "{input}",
                ),
                MessagesPlaceholder(
                    variable_name="agent_scratchpad"
                ),
            ]
        )

        self.agent = create_tool_calling_agent(
            self.llm,
            self.tools,
            self.prompt,
        )

        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
        )

        # conversation_id -> chat history
        self.conversations = {}

    def _get_history(
        self,
        conversation_id: str,
    ) -> InMemoryChatMessageHistory:

        if conversation_id not in self.conversations:

            self.conversations[
                conversation_id
            ] = InMemoryChatMessageHistory()

        return self.conversations[
            conversation_id
        ]

    def _get_sources(
        self,
        question: str,
    ) -> list[dict]:

        """
        Retrieve source documents and their
        similarity scores for the API response.
        """

        try:
            results = (
                vector_store_service.vector_store
                .similarity_search_with_score(
                    question,
                    k=TOP_K,
                )
            )

            sources = []

            for document, score in results:

                filename = document.metadata.get(
                    "filename"
                )

                if not filename:
                    filename = Path(
                        document.metadata.get(
                            "source",
                            "Unknown",
                        )
                    ).name

                page = document.metadata.get(
                    "page",
                    0,
                )

                sources.append(
                    {
                        "document": filename,
                        "page": int(page) + 1,
                        "relevance_score": float(
                            score
                        ),
                    }
                )

            return sources

        except Exception:
            return []

    def invoke(
        self,
        question: str,
        conversation_id: str,
    ) -> dict:

        history = self._get_history(
            conversation_id
        )

        response = self.executor.invoke(
            {
                "input": question,
                "history": history.messages,
            }
        )

        answer = response["output"]

        # Get structured source information
        sources = self._get_sources(
            question
        )

        # Store this turn
        history.add_user_message(
            question
        )

        history.add_ai_message(
            answer
        )

        return {
            "answer": answer,
            "sources": sources,
            "conversation_id": conversation_id,
        }


agent_service = AgentService()