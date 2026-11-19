import os
import io
import time
import hashlib
from pathlib import Path

import streamlit as st
import chromadb

from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from google.genai import types


# ==========================================
# 1. APP CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="DocSearch AI",
    page_icon="📚",
    layout="wide"
)

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

FAST_MODEL = os.getenv("FAST_MODEL", "gemini-3.8-flash")
DEEP_MODEL = os.getenv("DEEP_MODEL", "gemini-4-hydrogen")
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", FAST_MODEL)

EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "gemini-embedding-001"
)

VALID_THINKING_LEVELS = ["low", "medium", "high", "xhigh"]
DEFAULT_THINKING_LEVEL = os.getenv("THINKING_LEVEL", "low").lower()
if DEFAULT_THINKING_LEVEL not in VALID_THINKING_LEVELS:
    DEFAULT_THINKING_LEVEL = "low"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

if not API_KEY:
    st.error("GEMINI_API_KEY is missing in .env")
    st.stop()


# ==========================================
# 2. GEMINI CLIENT
# ==========================================

@st.cache_resource
def get_ai():
    return genai.Client(api_key=API_KEY)


ai = get_ai()


# ==========================================
# 3. CHROMADB PERSISTENT STORAGE
# ==========================================

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(
        path=str(DATA_DIR / "chroma_db")
    )

    collection = client.get_or_create_collection(
        name="docsearch_fast",
        metadata={"hnsw:space": "cosine"}
    )

    return collection


collection = get_collection()


# ==========================================
# 4. READ PDF AND TXT
# ==========================================

def read_document(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    pages = []

    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))

        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({
                    "page": page_no,
                    "text": text
                })

    elif uploaded_file.name.lower().endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="ignore")
        if text.strip():
            pages.append({
                "page": 1,
                "text": text
            })

    return pages


# ==========================================
# 5. TEXT CHUNKING
# ==========================================

def make_chunks(text, size=300, overlap=40):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


# ==========================================
# 6. GENERATE EMBEDDINGS
# ==========================================

def embed(texts, task):
    vectors = []
    batch_size = 50

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        result = ai.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task,
                output_dimensionality=768
            )
        )

        vectors.extend([
            item.values
            for item in result.embeddings
        ])

    return vectors


# ==========================================
# 7. PROCESS AND INDEX DOCUMENTS
# ==========================================

def build_index(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    existing = collection.get(
        where={"file_hash": file_hash},
        limit=1
    )

    if existing["ids"]:
        return {"status": "duplicate"}

    pages = read_document(uploaded_file)

    if not pages:
        return {"status": "empty"}

    documents = []
    metadatas = []
    ids = []

    for page in pages:
        chunks = make_chunks(page["text"])

        for index, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "filename": uploaded_file.name,
                "page": page["page"],
                "file_hash": file_hash
            })
            ids.append(
                f"{file_hash[:20]}_{page['page']}_{index}"
            )

    if not documents:
        return {"status": "empty"}

    vectors = embed(documents, "RETRIEVAL_DOCUMENT")

    for start in range(0, len(documents), 100):
        end = start + 100

        collection.add(
            documents=documents[start:end],
            embeddings=vectors[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end]
        )

    return {
        "status": "success",
        "chunks": len(documents)
    }


# ==========================================
# 8. CACHE QUESTION EMBEDDINGS
# ==========================================

@st.cache_data(ttl=3600, show_spinner=False)
def cached_query_embedding(question):
    result = ai.models.embed_content(
        model=EMBED_MODEL,
        contents=[question],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768
        )
    )

    return result.embeddings[0].values


# ==========================================
# 9. RETRIEVE RELEVANT DOCUMENTS
# ==========================================

def retrieve(question, top_k=4):
    total = collection.count()

    if total == 0:
        return [], []

    query_vector = cached_query_embedding(question)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, total),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results["documents"][0]
    sources = results["metadatas"][0]

    return documents, sources


# ==========================================
# 10. GEMINI INTERACTIONS API (FAST + RETRY)
# ==========================================

def _is_transient_503(error: Exception) -> bool:
    msg = str(error).lower()
    return "503" in msg or "unavailable" in msg or "high demand" in msg


def generate_answer(
    question,
    context,
    model_name=DEFAULT_CHAT_MODEL,
    thinking_level="low",
    max_retries=2
):
    prompt = f"""
You are DocSearch AI, a document question-answering assistant.

Answer the user's question using ONLY the context below.

IMPORTANT RULES:

1. Give accurate and clear answers.
2. Do not invent information.
3. If the answer is not in the documents, say:
   "This information was not found in the uploaded documents."
4. Use simple English.
5. Give a concise answer.
6. When comparing information, use all relevant context.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    candidate_models = [model_name]
    if model_name != FAST_MODEL:
        candidate_models.append(FAST_MODEL)

    last_error = None

    for current_model in candidate_models:
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": current_model,
                    "input": prompt,
                    "generation_config": {
                        "thinking_level": thinking_level
                    }
                }

                interaction = ai.interactions.create(**kwargs)
                answer = interaction.output_text
                return (
                    answer if answer else "No answer was generated.",
                    current_model
                )

            except Exception as error:
                last_error = error
                if _is_transient_503(error) and attempt < max_retries - 1:
                    time.sleep(1.5 ** attempt)
                    continue
                break

    raise last_error


# ==========================================
# 11. COMPLETE RAG ANSWER
# ==========================================

def answer_question(question, model_name=DEFAULT_CHAT_MODEL, thinking_level="low"):
    documents, sources = retrieve(question, top_k=4)

    if not documents:
        return (
            "Please upload and process a document first.",
            [],
            model_name
        )

    context_parts = []

    for document, source in zip(documents, sources):
        context_parts.append(
            f"""
Document: {source['filename']}

Page: {source['page']}

Content:
{document}
"""
        )

    context = "\n\n".join(context_parts)

    answer, used_model = generate_answer(
        question,
        context,
        model_name=model_name,
        thinking_level=thinking_level
    )

    return answer, sources, used_model


# ==========================================
# 12. STREAMLIT USER INTERFACE
# ==========================================

st.title("📚 DocSearch AI")

st.write(
    "Upload your PDF or TXT documents "
    "and ask questions using Gemini AI."
)


# ==========================================
# 13. SIDEBAR
# ==========================================

with st.sidebar:
    st.header("📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button(
        "🚀 Process Documents",
        use_container_width=True
    ):
        if not uploaded_files:
            st.warning("Please upload documents.")
        else:
            progress = st.progress(0)

            for index, file in enumerate(uploaded_files):
                try:
                    with st.spinner(f"Processing {file.name}..."):
                        result = build_index(file)

                    if result["status"] == "success":
                        st.success(f"{file.name} processed successfully!")
                        st.write("Chunks:", result["chunks"])
                    elif result["status"] == "duplicate":
                        st.info(f"{file.name} is already indexed.")
                    else:
                        st.warning(f"No readable text found in {file.name}.")

                except Exception as error:
                    st.error(f"Processing error: {error}")

                progress.progress((index + 1) / len(uploaded_files))

    st.divider()

    model_options = [FAST_MODEL, DEEP_MODEL]
    default_idx = model_options.index(DEFAULT_CHAT_MODEL) if DEFAULT_CHAT_MODEL in model_options else 0

    selected_model = st.selectbox(
        "Chat Model",
        options=model_options,
        index=default_idx,
        help="gemini-3.8-flash is fastest; gemini-4-hydrogen is for deeper reasoning."
    )

    thinking_level = st.selectbox(
        "Thinking Level",
        options=VALID_THINKING_LEVELS,
        index=VALID_THINKING_LEVELS.index(DEFAULT_THINKING_LEVEL),
        help="Use 'low' for fastest answers."
    )

    st.metric(
        "Indexed Chunks",
        collection.count()
    )

    st.caption(f"Default chat model: {DEFAULT_CHAT_MODEL}")
    st.caption(f"Embedding model: {EMBED_MODEL}")


# ==========================================
# 14. CHAT HISTORY
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("used_model"):
            st.caption(f"Answered by: {message['used_model']}")

        if message.get("sources"):
            with st.expander("📄 View Sources"):
                for source in message["sources"]:
                    st.write(
                        f"{source['filename']} "
                        f"| Page {source['page']}"
                    )


# ==========================================
# 15. CHAT INPUT
# ==========================================

question = st.chat_input(
    "Ask something about your documents..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                f"🔍 Searching documents and asking {selected_model}..."
            ):
                reply, sources, used_model = answer_question(
                    question,
                    model_name=selected_model,
                    thinking_level=thinking_level
                )

            st.markdown(reply)
            st.caption(f"Answered by: {used_model}")

            if sources:
                with st.expander("📄 View Sources"):
                    for source in sources:
                        st.write(
                            f"{source['filename']} "
                            f"| Page {source['page']}"
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": reply,
                "sources": sources,
                "used_model": used_model
            })

        except Exception as error:
            st.error(f"Gemini API Error: {error}")


# ==========================================
# 16. CLEAR CHAT
# ==========================================

if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()


st.divider()

st.caption(
    "DocSearch AI | Python | Streamlit | "
    "Gemini Interactions API | ChromaDB"
)