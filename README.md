# Universal Multimodal RAG System

A Python-based Retrieval-Augmented Generation (RAG) application for asking questions about information from multiple source types: documents, images, YouTube videos, and public GitHub repositories. The application uses a Streamlit interface and can generate answers from retrieved source content with supporting citations or source metadata where available.

## Features

- **Document RAG:** Ingest supported PDF, DOCX, PPTX, TXT, and CSV files, extract text, split content into chunks, create embeddings, and retrieve relevant passages for questions.
- **Image understanding:** Process images using OCR and a vision model, then index the extracted or described content for retrieval.
- **YouTube RAG:** Retrieve transcript content from supported YouTube videos and answer questions using transcript evidence, with timestamp metadata when available.
- **GitHub repository analysis:** Process public repositories, inspect files and code, and answer questions about repository structure and implementation.
- **Semantic retrieval:** Use Sentence Transformers embeddings and ChromaDB for vector search.
- **Source-aware responses:** Keep retrieval scoped to a source type and, where configured, the selected file or source identifier to reduce cross-source results.
- **Citations and metadata:** Display available document page, file, YouTube timestamp, or GitHub code location information.
- **Multiple LLM providers:** Configure local Ollama or supported cloud providers through environment variables.
- **Streamlit UI:** Choose a source and interact with its processing and question-answering workspace.

> **Note:** Functionality depends on the source content, installed system tools, available model, and configured API credentials. OCR quality and response time vary by image and model.

## Technology Stack

- **Language:** Python
- **UI:** Streamlit
- **RAG and retrieval:** LangChain components, Sentence Transformers, ChromaDB
- **Document processing:** PyMuPDF, python-docx, python-pptx, pandas
- **Image processing:** Pillow, pytesseract, Tesseract OCR, Ollama vision model (for example, `llava:7b`)
- **YouTube transcripts:** `youtube-transcript-api`
- **LLM integration:** Ollama and configured cloud providers
- **Metadata storage:** SQLite

## Architecture

```text
User
  |
  v
Streamlit UI
  |
  v
Source-specific ingestion
  |-- Documents: extraction -> cleaning -> chunking
  |-- Images: preprocessing -> OCR / vision description
  |-- YouTube: transcript extraction -> timestamped chunks
  |-- GitHub: repository inspection -> code parsing / chunking
  |
  v
Metadata + embeddings
  |
  v
ChromaDB vector retrieval
  |
  v
Source-aware retrieval / context preparation
  |
  v
Configured LLM
  |
  v
Answer, summary, generated questions, and source metadata
```

## Project Structure

```text
.
├── app.py                  # Streamlit application
├── backend/
│   ├── embeddings/          # Embedding and indexing components
│   ├── ingestion/           # Source loaders and ingestion pipelines
│   ├── models/              # LLM and vision model integrations
│   ├── processing/          # Text/image processing and chunking
│   ├── rag/                 # Retrieval, generation, tasks, and citations
│   ├── utils/               # Shared helpers and metadata utilities
│   └── vectorstore/         # ChromaDB integrations
├── requirements.txt         # Python dependencies
├── .gitignore               # Excluded files and local data
└── README.md
```

The exact contents of `backend/` may evolve as components are updated.

## Requirements

- Python 3.12 is the development version used for this project.
- Git
- An Ollama installation if using local Ollama models
- Tesseract OCR installed and available to `pytesseract` for local OCR
- API credentials for whichever cloud LLM provider you choose

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ashokkm07/Multimodel-rag-system-.git
cd Multimodel-rag-system-
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a dependency fails to install, check that your Python version is compatible with the pinned package versions in `requirements.txt`.

### 4. Configure environment variables

Create a local `.env` file in the project root. Do not commit this file or publish API keys.

Example configuration (leave unused provider keys blank):

```dotenv
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:3b

OPENROUTER_API_KEY=
OPENROUTER_MODEL=

GEMINI_API_KEY=
GEMINI_MODEL=

OPENAI_API_KEY=
OPENAI_MODEL=
```

Use the provider names and model identifiers supported by the application's provider configuration. Model availability and access depend on your local installation or provider account.

### 5. Configure Ollama (for local inference)

Install Ollama, start its service, and download the text model:

```bash
ollama pull llama3.2:3b
```

For image understanding, if the image pipeline is configured to use LLaVA, download it as well:

```bash
ollama pull llava:7b
```

Set the local provider in `.env`:

```dotenv
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:3b
```

### 6. Install Tesseract OCR

Install Tesseract OCR for your operating system and ensure the executable is available to `pytesseract`. On Windows, a common location is:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

The application may configure this standard location automatically. If Tesseract is installed elsewhere, configure the path in the application or add it to your system `PATH`.

## Run the Application

From the project root with the virtual environment activated:

```bash
streamlit run app.py --server.fileWatcherType none
```

The `--server.fileWatcherType none` option can help avoid optional-module import errors caused by automatic file watching in some environments.

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser.

## Usage

1. Select **Document**, **Image**, **YouTube link**, or **GitHub repo link** in the sidebar.
2. Provide a file or link for the selected source.
3. Run its processing action to extract content and index it.
4. Ask a question or use available summary / question-generation actions.
5. Review the displayed source metadata and citations to check which retrieved content supports the response.

For GitHub ingestion, use a public repository URL unless private-repository access has been explicitly configured.

## Data and Security

- Keep `.env` private. Never commit real API keys, tokens, passwords, or credentials.
- Do not commit uploaded documents or images, local databases, vector-store directories, virtual environments, model weights, or private data.
- Check `git status` and review staged files before every push.
- If a secret has ever been pushed to a repository, rotate or revoke it; deleting the file in a later commit does not remove it from Git history.

## Troubleshooting

- **Cloud provider authentication errors:** Check the corresponding API key, model identifier, account permissions, and provider quota.
- **Rate limits or service unavailable:** Retry later or switch to another configured provider.
- **Ollama connection errors:** Confirm that Ollama is installed, running, and that the requested model has been pulled.
- **Image OCR returns no text:** Verify that Tesseract is installed, that the language data is available, and that the image is clear enough for OCR. Images can still be processed by the configured vision path when available.
- **No source results:** Process/index the intended source first and verify that retrieval is scoped to the same source identifier used during indexing.
- **Slow YouTube same-language responses:** Generation time depends on the local model and transcript context size; try a smaller context or a faster model when configured.

## Current Scope

This repository combines source-specific ingestion pipelines with a common retrieval and generation interface. Results are dependent on the quality of extracted content and the selected LLM. Always inspect citations and source metadata when validating an answer. Cloud-provider integrations require valid credentials and may be affected by service availability or usage limits.

## License

No license is specified in this repository yet. Add a `LICENSE` file if you intend to publish the project for reuse by others.
