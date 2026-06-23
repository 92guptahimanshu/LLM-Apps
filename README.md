# LLM-Apps
This repository contains various applications built using Large Language Models (LLMs).

## Text Summarization (Streamlit + LangChain + Groq)

A small Streamlit app that utilizes LangChain and Groq to summarize the textual content of a web page or YouTube video. The app supports chunking and map-reduce summarization to avoid model token limits.

### Features
- Summarize website pages or YouTube videos.
- Utilizes Groq `llama-3.1-8b-instant` (configurable) via `langchain-groq`.
- Implements chunking (using `RecursiveCharacterTextSplitter`) and map-reduce summarization to minimize TPM/token errors.
- User-friendly sidebar UI to supply the Groq API key or fallback to `.env`.

### Requirements
- Python 3.10+ (3.11 recommended).
- A virtual environment.
- Install dependencies:

```
python -m pip install -r "Text Summarization/requirements.txt"
```

### Environment Variables
Create a `.env` file in the repository root or in `Text Summarization/` with the following keys (do not commit this file):

```
GROQ_API_KEY=<your_groq_api_key>
# Optional (if you use Hugging Face models or private gated repos)
HUGGINGFACE_HUB_TOKEN=<hf_token>
```

The app will prioritize a key entered into the Streamlit sidebar and will fall back to a value found in environment variables.

### Run the App
Activate your virtual environment, then run Streamlit from the repository root:

**Windows (PowerShell):**

```
.\.venv\Scripts\Activate.ps1
python -m streamlit run "Text Summarization/app.py"
```

**macOS / Linux:**

```
source .venv/bin/activate
python -m streamlit run "Text Summarization/app.py"
```

### Usage
1. Open the app in your browser when Streamlit starts.
2. Provide a Groq API key in the sidebar or ensure `GROQ_API_KEY` is set in `.env`.
3. Paste a website or YouTube URL and click `Summarize the Content from YT or Website`.

### Troubleshooting
- **"Request too large" / 413 TPM errors:** Reduce `SPLIT_CHUNK_SIZE`, decrease `BATCH_SIZE`, add more throttling (increase sleep), or use a smaller model / upgrade service tier.
- **HuggingFace 401 / expired token:** Regenerate a token at [Hugging Face Tokens](https://huggingface.co/settings/tokens) and update `HUGGINGFACE_HUB_TOKEN` or run `huggingface-cli login`.
- **Permission denied on `.vs` files while committing:** Close Visual Studio, add `.vs/` to `.gitignore`, then run `git rm -r --cached .vs`.

### Security / Git
- The `.gitignore` file includes `.env` and `Text Summarization/.env`. Do not commit secrets.
- If secrets were accidentally pushed, rotate them immediately and consider purging git history (using BFG or `git filter-repo`).
- To stop tracking already-added env files:

```
git rm --cached .env "Text Summarization/.env" || true
git commit -m "chore: stop tracking .env files"
git push
```

### Contributing
- Open an issue or submit a pull request (PR). Keep changes small and include tests or usage notes where appropriate.

### License
This project is licensed under the MIT License.

---

## AI Data Quality (Streamlit + Groq)

A Streamlit application that automates common data-quality checks (completeness, validity, uniqueness, and accuracy) and can auto-suggest validation rules using the Groq LLM API.

### Features
- Inspect CSV datasets for completeness, validity, and uniqueness.
- Apply manual and AI-suggested rules for accuracy checks.
- Generate, preview, and download detailed DQ result CSVs and failed-record reports.
- Integrates with Groq LLM to convert plain-English rules into executable JSON rules.

### Requirements
- Python 3.10+ (3.11 recommended).
- Install dependencies (from repo root):

```
python -m pip install -r "LLM_DQ/requirements.txt"
```

### Environment Variables
- `GROQ_API_KEY` — your Groq API key (or provide in the app sidebar).

### Run the App

**Windows (PowerShell):**

```
python -m streamlit run "LLM_DQ/app.py"
```

**macOS / Linux:**

```
python -m streamlit run "LLM_DQ/app.py"
```

### Usage
1. Open the app in your browser when Streamlit starts.
2. Provide your Groq API key in the sidebar (or set `GROQ_API_KEY` in `.env`).
3. Upload a CSV and configure the columns you want checked (completeness, validity, uniqueness, accuracy).
4. (Optional) Enter plain-English accuracy rules or use AI to auto-suggest rules.
5. Run the checks and download the DQ results or failed records.

### Troubleshooting
- If the Groq LLM call fails, verify your API key and network connectivity.
- If AI output cannot be parsed to JSON, inspect raw LLM output shown in the UI and try reducing prompt complexity.

---

## CodeAssistant (Gradio Local Proxy)

A lightweight Gradio front-end that forwards user prompts to a locally running model API (or other HTTP-based model endpoints). Useful for quick UI-driven testing of local LLM servers.

### Features
- Simple chat input box and streamed output (depends on backend).
- Designed to call a localhost inference endpoint (configurable URL and port).
- Minimal dependency set (requests + gradio).

### Requirements
- Python 3.10+.
- Start your local model API (example: a local inference server on port 11434).
- Install Python dependencies:

```
python -m pip install -r "CodeAssistant/requirements.txt"
```

### Run the App

```
python "CodeAssistant/app.py"
```

### Usage
1. Ensure your local model API is running and reachable (default: `http://localhost:11434/api/generate`).
2. Launch the Gradio UI; enter text prompts and get responses from your local inference endpoint.

### Security
- The app assumes a locally-hosted inference service; do not expose the local endpoint to the public without proper authentication.

### Troubleshooting
- If you get connection errors, confirm the local server URL and port, and that CORS or firewall rules aren’t blocking requests.
- Check the model server logs for request details and errors.

---

### Changes Made:
1. Added a brief introduction to the repository at the beginning.
2. Ensured consistent formatting and clarity throughout the document.
3. Enhanced the flow of information by maintaining a logical structure.
4. Added minor clarifications and formatting improvements for better readability.
