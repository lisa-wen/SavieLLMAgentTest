from dotenv import load_dotenv
import os, yaml
from uuid import uuid4
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Environment & API Key ---
load_dotenv()

# --- Pfade & Parameter ---
FAQ_DIR     = "documents/FAQ"
PERSIST_DIR = "./chroma_langchain_db"
COLLECTION  = "example_collection"

# Mehrsprachigkeit
languages = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "pl": "Polish",
}

# --- LLM & Embeddings ---
llm        = ChatOpenAI(temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- Chains definieren ---
translation_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["text","target_language"],
        template="Übersetze ins {target_language}: {text}"
    )
)
paraphrase_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["question","language"],
        template="""
Formuliere 3 alternative, gängige Fragestellungen auf {language} zu:
{question}
Gib jede Paraphrase in einer neuen Zeile aus.
"""
    )
)

# --- Einlesen & Paraphrasieren ---
docs = []
for fname in os.listdir(FAQ_DIR):
    if not fname.endswith(".txt"):
        continue
    raw = open(os.path.join(FAQ_DIR, fname), encoding="utf-8", errors="ignore").read().strip()

    # Frontmatter (optional)
    metadata = {}
    if raw.startswith("---"):
        _, fm, body = raw.split("---", 2)
        metadata.update(yaml.safe_load(fm))
        if "tags" in metadata and isinstance(metadata["tags"], list):
            metadata["tags"] = ", ".join(metadata["tags"])
    else:
        body = raw

    # Frage/Antwort extrahieren
    lines = body.splitlines()
    if lines and lines[0].startswith("Q:"):
        question = lines[0][2:].strip()
        answer   = "\n".join(lines[1:]).strip()
    elif lines and lines[0].endswith("?"):
        question = lines[0].strip()
        answer   = "\n".join(lines[1:]).strip()
    else:
        parts    = body.split("\n\n", 1)
        question = parts[0].strip()
        answer   = parts[1].strip() if len(parts)>1 else ""

    # Varianten pro Sprache
    for code, name in languages.items():
        translation = (
            translation_chain.run(text=question, target_language=name).strip()
            if code != "de" else question
        )
        para_text = paraphrase_chain.run(question=translation, language=name)
        variants  = [translation] + [p.strip("- ").strip() for p in para_text.splitlines() if p.strip()]

        for variant in variants:
            content = f"Q: {variant}\nA: {answer}"
            doc_meta = {"id": question, "paraphrase": variant, "language": code}
            doc_meta.update(metadata)
            docs.append(Document(page_content=content, metadata=doc_meta, id=str(uuid4())))

# --- Neuer Flat-Index via from_documents() ---
# 1) Verzeichnis ./chroma_langchain_db VORHER löschen!
store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION,
    # Flat-Index aktivieren
    collection_metadata={"index_factory": "flat"}
)

print(f"Ingested {len(docs)} FAQ-Varianten in Collection '{COLLECTION}'.")
