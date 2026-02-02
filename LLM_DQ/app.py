import validators
import streamlit as st
from time import sleep

from langchain_community.document_loaders import (
    YoutubeLoader,
    UnstructuredURLLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(
    page_title="Summarize YT / Website",
    page_icon="🦜",
)

st.title("🦜 LangChain: Summarize YouTube or Website")
st.caption("Groq • Chunked • Map-Reduce • Streaming")

with st.sidebar:
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
    )

url = st.text_input("Enter YouTube or Website URL")

# -----------------------------
# LLM Setup (Groq)
# -----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key,
    temperature=0.5,
)

# -----------------------------
# Prompts
# -----------------------------
map_prompt = ChatPromptTemplate.from_template("""
Summarize the following chunk concisely:

{text}
""")

reduce_prompt = ChatPromptTemplate.from_template("""
You are given multiple partial summaries.
Combine them into a clear, coherent final summary (~300 words).

Summaries:
{text}
""")

# -----------------------------
# Safe Invoke (Groq is stable, but kept for safety)
# -----------------------------
def safe_invoke(chain, payload, retries=2):
    for _ in range(retries):
        try:
            return chain.invoke(payload)
        except Exception:
            sleep(0.5)
    raise RuntimeError("LLM failed after retries")

# -----------------------------
# Button Action
# -----------------------------
if st.button("Summarize"):
    if not groq_api_key.strip() or not url.strip():
        st.error("Please provide both the Groq API key and a URL.")
    elif not validators.url(url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Loading and summarizing content..."):
                # -------- Load Content --------
                if "youtube.com" in url or "youtu.be" in url:
                    loader = YoutubeLoader.from_youtube_url(
                        url,
                        add_video_info=True,
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"},
                    )

                docs = loader.load()

                # -------- Split into chunks --------
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=200,
                )
                chunks = splitter.split_documents(docs)

                # -------- MAP STEP --------
                map_chain = map_prompt | llm
                partial_summaries = []

                for chunk in chunks:
                    if not chunk.page_content.strip():
                        continue

                    result = safe_invoke(
                        map_chain,
                        {"text": chunk.page_content}
                    )

                    partial_summaries.append(result.content)

                if not partial_summaries:
                    st.error("No content could be summarized.")
                    st.stop()

                combined_summary = "\n\n".join(partial_summaries)

                # -------- REDUCE STEP (Streaming) --------
                reduce_chain = reduce_prompt | llm

                placeholder = st.empty()
                final_text = ""

                for token in reduce_chain.stream(
                    {"text": combined_summary}
                ):
                    final_text += token.content
                    placeholder.markdown(final_text)

        except Exception as e:
            st.exception(e)
