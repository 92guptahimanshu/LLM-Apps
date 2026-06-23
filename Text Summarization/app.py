import os
import validators
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

# Load environment variables (tries project root and local folder)
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

## streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

## Get the Groq API Key and url (YT or website) to be summarized
with st.sidebar:
    groq_api_key_input = st.text_input("Groq API Key", value="", type="password")

# Try environment keys (support both upper/lower cases that might appear in .env)
groq_api_key_env = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key") or ""
groq_api_key_env = groq_api_key_env.strip().strip('\'"') if groq_api_key_env else ""
groq_api_key = groq_api_key_input.strip() if groq_api_key_input.strip() else groq_api_key_env

if not groq_api_key:
    with st.sidebar:
        st.warning("Provide Groq API key in sidebar or set GROQ_API_KEY / groq_api_key in .env")

generic_url = st.text_input("URL", label_visibility="collapsed")

# Prompts: map + reduce for map_reduce chain (map_reduce requires separate prompts)
map_prompt = PromptTemplate(
    template="""
Summarize the following chunk concisely and keep the key facts and intent intact:

{text}
""",
    input_variables=["text"],
)

reduce_prompt = PromptTemplate(
    template="""
You are given multiple partial summaries. Combine them into a clear, coherent final summary (~300 words) preserving important facts and removing duplication.

Summaries:
{text}
""",
    input_variables=["text"],
)

# A generic fallback prompt (kept for 'stuff' chain if needed)
prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Text splitter config - reduce chunk_size if you still hit token limits
SPLIT_CHUNK_SIZE = 1500
SPLIT_CHUNK_OVERLAP = 200

if st.button("Summarize the Content from YT or Website"):
    if not groq_api_key:
        st.error("Please provide a Groq API key in the sidebar or set GROQ_API_KEY / groq_api_key in your .env.")
    elif not generic_url.strip():
        st.error("Please provide the URL to summarize.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or website URL.")
    else:
        try:
            with st.spinner("Downloading content and preparing summary..."):
                # load source documents
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=True,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        },
                    )
                docs = loader.load()

                # split long documents into smaller chunks to avoid token limits / TPM bursts
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=SPLIT_CHUNK_SIZE, chunk_overlap=SPLIT_CHUNK_OVERLAP)
                split_docs = text_splitter.split_documents(docs)

                # create llm client after we have a validated key
                llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

                # Use map_reduce with explicit map_prompt and combine_prompt to avoid "extra inputs" validation error
                chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",
                    map_prompt=map_prompt,
                    combine_prompt=reduce_prompt,
                )

                # Batch the chunks to reduce TPM spikes; LangChain map_reduce will call map and then reduce
                BATCH_SIZE = 4
                combined_results = []
                for i in range(0, len(split_docs), BATCH_SIZE):
                    batch = split_docs[i : i + BATCH_SIZE]
                    batch_summary = chain.run(batch)
                    combined_results.append(batch_summary)
                    # throttle to avoid TPM spikes; increase sleep if you still hit limits
                    time.sleep(1)

                # Combine batch summaries into a final summary using the same reduce prompt chain
                final_input_docs = [{"page_content": s} for s in combined_results]
                final_summary = chain.run(final_input_docs)

                st.success(final_summary)

        except Exception as e:
            # Provide more actionable guidance when the model returns a 413 TPM/size error
            err_text = str(e)
            if "Request too large" in err_text or "413" in err_text or "tokens per minute" in err_text:
                st.error(
                    "Request too large for the chosen model / tier. Actions:\n"
                    "- Reduce chunk size (SPLIT_CHUNK_SIZE) or chunk overlap.\n"
                    "- Lower BATCH_SIZE or add longer throttling (increase time.sleep).\n"
                    "- Use a smaller model or upgrade your Groq service tier.\n"
                    "- Try summarizing a shorter URL or fewer sections at once."
                )
            else:
                st.exception(f"Exception: {e}")