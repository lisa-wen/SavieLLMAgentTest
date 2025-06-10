__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional, Dict, Any
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

class FaqToolInput(BaseModel):
    query: str = Field(..., description="The user question")
    k: Optional[int] = Field(2, description="Number of top documents to retrieve")
    language: Optional[str] = Field(None, description="ISO code of language to filter, e.g. 'de', 'en'")
    category: Optional[str] = Field(None, description="Category tag to filter by")

class FaqTool(BaseTool):
    name: str = "faq_tool"
    description: str = (
        "Beantworte Fragen anhand der FAQ-Dokumente in der gewählten Sprache und Kategorie."
    )
    args_schema: Type[BaseModel] = FaqToolInput

    def __init__(self, persist_directory: str, collection_name: str, prompt_template: str):
        super().__init__()
        # LLM und Embeddings
        self._llm = ChatOpenAI(temperature=0)
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        # Chroma-Store (mit Standard-Einstellungen)
        self._store = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=self._embeddings
        )
        # Prompt-Template
        self._prompt_template = prompt_template

    def _configure_chain(self, language: Optional[str], category: Optional[str], k: int) -> None:
        # Filter nach Sprache/Kategorie
        filters: Dict[str, str] = {
            **({"language": language} if language else {}),
            **({"category": category} if category else {})
        }
        retriever = self._store.as_retriever(search_kwargs={
            "k": k,
            "filter": filters
        })
        # Prompt nur mit context & question, Sprache/Kategorie als Literal
        suffix = f"(Kategorie: {category or 'alle'}, Sprache: {language or 'alle'})"
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self._prompt_template + "\n\n" + suffix
        )
        self._qa_chain = RetrievalQA.from_chain_type(
            llm=self._llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def _run(
        self,
        query: str,
        k: int = 2,
        language: Optional[str] = None,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        # Chain konfigurieren
        self._configure_chain(language, category, k)
        # RetrievalQA erwartet nur 'query'
        result = self._qa_chain({"query": query})
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }

    async def _arun(
        self,
        query: str,
        k: int = 2,
        language: Optional[str] = None,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        self._configure_chain(language, category, k)
        result = await self._qa_chain.arun({"query": query})
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }
