"""LangChain LCEL chain for context-aware response generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from src.common.config import Settings

logger = structlog.get_logger()

RAG_PROMPT_TEMPLATE = """\
You are a knowledgeable assistant. Answer the user's question based on \
the provided context. If the context doesn't contain enough information \
to answer, say so clearly. Always cite which source(s) you used.

Context:
{context}

Question: {question}

Answer:"""


class GenerationChain:
    """Builds and runs the RAG generation chain using LCEL.

    Args:
        settings: Application settings for LLM provider and model selection.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = self._create_llm()
        self._prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self._chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self._prompt
            | self._llm
            | StrOutputParser()
        )

    def _create_llm(self) -> BaseChatModel:
        """Create the LLM based on provider setting.

        Returns:
            A LangChain chat model instance.
        """
        if self._settings.llm_provider == "ollama":
            return ChatOllama(
                model=self._settings.llm_model,
                base_url=self._settings.ollama_base_url,
            )
        return ChatOpenAI(
            model=self._settings.llm_model,
            openai_api_key=self._settings.openai_api_key,
            temperature=0.1,
        )

    async def generate(self, context: str, question: str) -> str:
        """Generate a response using the RAG chain.

        Args:
            context: Formatted context from retrieved documents.
            question: The user's question.

        Returns:
            The LLM-generated answer.
        """
        logger.info("generation_started", question_length=len(question))

        response = await self._chain.ainvoke({"context": context, "question": question})

        logger.info("generation_completed", response_length=len(response))
        return response
