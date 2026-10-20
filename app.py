import os
import sys
import importlib
import hashlib
import sqlite3
import inspect
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# ==============================================================
# PROJECT SETUP
# ==============================================================

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================
# TESSERACT CONFIGURATION
# ==============================================================

def configure_tesseract():
    try:
        import pytesseract

        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]

        for path in possible_paths:
            if Path(path).exists():
                pytesseract.pytesseract.tesseract_cmd = path
                return True

    except Exception:
        pass

    return False


configure_tesseract()


# ==============================================================
# BACKEND IMPORTS
# ==============================================================

try:
    from backend.models.llm_service import LLMService

    from backend.rag.multimodal_generation import (
        MultimodalRAGGenerator,
    )

    from backend.rag.multimodal_tasks import (
        MultimodalTasks,
    )

    from backend.ingestion.image_ingestion_pipeline import (
        ImageIngestionPipeline,
    )

    from backend.embeddings.image_indexer import (
        ImageIndexer,
    )

    IMPORT_BOOTSTRAP_ERROR = None

except Exception as error:
    LLMService = None
    MultimodalRAGGenerator = None
    MultimodalTasks = None
    ImageIngestionPipeline = None
    ImageIndexer = None

    IMPORT_BOOTSTRAP_ERROR = error


# ==============================================================
# PAGE CONFIG
# ==============================================================

st.set_page_config(
    page_title="Universal Multimodal RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================
# CSS
# ==============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1rem;
        color: #666666;
        margin-bottom: 1.5rem;
    }

    .answer-box {
        border: 1px solid #dddddd;
        border-radius: 10px;
        padding: 16px;
        margin-top: 10px;
    }

    .citation-card {
        border-left: 4px solid #666666;
        padding: 10px 14px;
        margin: 8px 0;
        background: #f7f7f7;
        border-radius: 6px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================
# SESSION STATE
# ==============================================================

DEFAULT_STATE = {
    "processed": False,
    "processed_source": None,
    "processed_name": None,
    "processed_path": None,
    "processed_retrieval_source": None,
    "processed_result": None,

    "selected_file_hash": None,
    "processed_file_hash": None,

    "last_output_type": None,
    "last_question": None,
    "last_answer": None,
    "last_summary": None,
    "last_questions": [],
    "last_sources": [],

    "active_source": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==============================================================
# OUTPUT RESET
# ==============================================================

def clear_outputs():
    st.session_state.last_output_type = None
    st.session_state.last_question = None
    st.session_state.last_answer = None
    st.session_state.last_summary = None
    st.session_state.last_questions = []
    st.session_state.last_sources = []


def reset_processing():
    st.session_state.processed = False
    st.session_state.processed_source = None
    st.session_state.processed_name = None
    st.session_state.processed_path = None
    st.session_state.processed_retrieval_source = None
    st.session_state.processed_result = None

    st.session_state.selected_file_hash = None
    st.session_state.processed_file_hash = None

    clear_outputs()


# ==============================================================
# FILE HASH
# ==============================================================

def get_uploaded_file_hash(uploaded_file):
    content = uploaded_file.getvalue()

    identity = (
        uploaded_file.name.encode("utf-8")
        + b"\x00"
        + content
    )

    return hashlib.sha256(
        identity
    ).hexdigest()


# ==============================================================
# UPLOAD
# ==============================================================

def save_uploaded_file(uploaded_file):
    upload_dir = (
        PROJECT_ROOT
        / "data"
        / "uploads"
    )

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_name = Path(
        uploaded_file.name
    ).name

    file_path = (
        upload_dir
        / safe_name
    )

    with file_path.open("wb") as file:
        file.write(
            uploaded_file.getbuffer()
        )

    return file_path


def get_relative_source(file_path):
    try:
        relative_path = (
            Path(file_path)
            .resolve()
            .relative_to(
                PROJECT_ROOT.resolve()
            )
        )

        return str(
            relative_path
        )

    except ValueError:
        return str(file_path)


# ==============================================================
# LLM CONFIGURATION
# ==============================================================

def get_llm_config():
    provider = st.session_state.get(
        "llm_provider",
        "ollama",
    )

    model = st.session_state.get(
        "llm_model",
        "llama3.2:3b",
    )

    return provider, model


def get_llm_service():
    provider, model = get_llm_config()

    if LLMService is None:
        raise RuntimeError(
            "LLMService import failed: "
            f"{IMPORT_BOOTSTRAP_ERROR}"
        )

    return LLMService(
        provider=provider,
        model_name=model,
    )


def get_multimodal_generator():
    provider, model = get_llm_config()

    if MultimodalRAGGenerator is None:
        raise RuntimeError(
            "MultimodalRAGGenerator import failed: "
            f"{IMPORT_BOOTSTRAP_ERROR}"
        )

    return MultimodalRAGGenerator(
        top_k=5,
        provider=provider,
        model_name=model,
    )


def get_multimodal_tasks():
    provider, model = get_llm_config()

    if MultimodalTasks is None:
        raise RuntimeError(
            "MultimodalTasks import failed: "
            f"{IMPORT_BOOTSTRAP_ERROR}"
        )

    return MultimodalTasks(
        top_k=5,
        provider=provider,
        model_name=model,
    )


# ==============================================================
# SOURCE LABEL
# ==============================================================

def format_source_type(source_type):
    labels = {
        "document": "Document",
        "image": "Image",
        "youtube": "YouTube",
        "github": "GitHub",
    }

    return labels.get(
        source_type,
        str(source_type).title(),
    )


# ==============================================================
# CITATIONS
# ==============================================================

def render_citations(sources):
    if not sources:
        return

    st.subheader(
        "Sources & Citations"
    )

    for index, source in enumerate(
        sources,
        start=1,
    ):
        source_type = source.get(
            "source_type",
            "unknown",
        )

        filename = source.get(
            "filename"
        )

        source_name = source.get(
            "source"
        )

        page = source.get(
            "page"
        )

        timestamp = source.get(
            "timestamp"
        )

        start_time = source.get(
            "start_time"
        )

        end_time = source.get(
            "end_time"
        )

        chunk_index = source.get(
            "chunk_index"
        )

        start_line = source.get(
            "start_line"
        )

        end_line = source.get(
            "end_line"
        )

        function_name = source.get(
            "function"
        )

        class_name = source.get(
            "class_name"
        )

        st.markdown(
            '<div class="citation-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"**[{index}] "
            f"{format_source_type(source_type)}**"
        )

        if filename:
            st.write(
                f"📄 File: {filename}"
            )

        elif source_name:
            st.write(
                f"📄 Source: {source_name}"
            )

        if class_name:
            st.write(
                f"🏷️ Class: {class_name}"
            )

        if function_name:
            st.write(
                f"🔧 Function: {function_name}"
            )

        if page is not None:
            st.write(
                f"📖 Page: {page}"
            )

        if timestamp:
            st.write(
                f"▶️ Timestamp: {timestamp}"
            )

        elif start_time is not None:

            if end_time is not None:
                st.write(
                    f"▶️ Time: "
                    f"{start_time}s – "
                    f"{end_time}s"
                )
            else:
                st.write(
                    f"▶️ Time: "
                    f"{start_time}s"
                )

        if chunk_index is not None:
            st.write(
                f"🧩 Chunk: {chunk_index}"
            )

        if (
            start_line is not None
            or end_line is not None
        ):

            if (
                start_line is not None
                and end_line is not None
            ):
                st.write(
                    f"📍 Lines: "
                    f"{start_line}–{end_line}"
                )

            elif start_line is not None:
                st.write(
                    f"📍 Line: "
                    f"{start_line}"
                )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


# ==============================================================
# GENERIC IMPORT
# ==============================================================

def import_class(candidates):
    errors = []

    for module_name, class_name in candidates:

        try:
            module = importlib.import_module(
                module_name
            )

            if hasattr(
                module,
                class_name,
            ):
                return getattr(
                    module,
                    class_name,
                )

        except Exception as error:
            errors.append(
                f"{module_name}.{class_name}: "
                f"{error}"
            )

    raise ImportError(
        "Could not load required backend class.\n"
        + "\n".join(errors)
    )


# ==============================================================
# GENERIC METHOD CALL
# ==============================================================

def call_compatible(
    obj,
    method_names,
    *args,
    **kwargs,
):
    for method_name in method_names:

        if not hasattr(
            obj,
            method_name,
        ):
            continue

        method = getattr(
            obj,
            method_name,
        )

        try:
            return method(
                *args,
                **kwargs,
            )

        except TypeError:
            try:
                return method(
                    *args,
                )
            except TypeError:
                continue

    raise AttributeError(
        "None of the expected methods were found: "
        + ", ".join(method_names)
    )


# ==============================================================
# DOCUMENT
# ==============================================================

def process_document(file_path):

    DocumentPipeline = import_class(
        [
            (
                "backend.ingestion.ingestion_pipeline",
                "IngestionPipeline",
            ),
            (
                "backend.ingestion.ingestion_pipeline",
                "DocumentIngestionPipeline",
            ),
        ]
    )

    pipeline = DocumentPipeline()

    return call_compatible(
        pipeline,
        [
            "process",
            "ingest",
            "run",
        ],
        str(file_path),
    )


# ==============================================================
# YOUTUBE
# ==============================================================

def process_youtube(youtube_url):

    YouTubePipeline = import_class(
        [
            (
                "backend.ingestion.youtube_ingestion_pipeline",
                "YouTubeIngestionPipeline",
            ),
        ]
    )

    pipeline = YouTubePipeline()

    return call_compatible(
        pipeline,
        [
            "process",
            "ingest",
            "run",
        ],
        youtube_url,
    )


# ==============================================================
# GITHUB
# ==============================================================

def process_github(github_url):

    GitHubPipeline = import_class(
        [
            (
                "backend.ingestion.github_repository_pipeline",
                "GitHubRepositoryPipeline",
            ),
            (
                "backend.ingestion.github_pipeline",
                "GitHubPipeline",
            ),
        ]
    )

    pipeline = GitHubPipeline()

    return call_compatible(
        pipeline,
        [
            "process",
            "ingest",
            "run",
        ],
        github_url,
    )


# ==============================================================
# GITHUB URL NORMALIZATION
# ==============================================================

def normalize_github_repo_key(url):
    """
    Converts different GitHub URL formats into:

        owner/repository

    Example:

        https://github.com/Ashokkm07/dog-breed-classification-
        git@github.com:Ashokkm07/dog-breed-classification-.git

    become:

        ashokkm07/dog-breed-classification-
    """

    if not url:
        return ""

    value = str(
        url
    ).strip().lower()

    value = value.replace(
        "\\",
        "/",
    )

    if value.startswith(
        "git@github.com:"
    ):
        value = value[
            len("git@github.com:") :
        ]

    if value.startswith(
        "https://"
    ):
        value = value[
            len("https://") :
        ]

    elif value.startswith(
        "http://"
    ):
        value = value[
            len("http://") :
        ]

    if value.startswith(
        "www."
    ):
        value = value[
            len("www.") :
        ]

    if value.startswith(
        "github.com/"
    ):
        value = value[
            len("github.com/") :
        ]

    value = value.split(
        "?",
        1,
    )[0]

    value = value.split(
        "#",
        1,
    )[0]

    value = value.rstrip(
        "/"
    )

    if value.endswith(
        ".git"
    ):
        value = value[
            :-4
        ]

    parts = [
        part
        for part in value.split("/")
        if part
    ]

    if len(parts) >= 2:
        return (
            parts[-2]
            + "/"
            + parts[-1]
        )

    return value


# ==============================================================
# GET GITHUB REPOSITORY ID
# ==============================================================

def get_github_repository_id(
    github_url
):
    """
    Finds the repository ID in the existing SQLite database.
    """

    db_path = (
        PROJECT_ROOT
        / "data"
        / "rag_metadata.db"
    )

    if not db_path.exists():
        raise RuntimeError(
            "GitHub metadata database was not found:\n"
            f"{db_path}"
        )

    connection = sqlite3.connect(
        str(db_path)
    )

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'github_repositories'
            """
        )

        if cursor.fetchone() is None:
            raise RuntimeError(
                "github_repositories table was not found "
                "in rag_metadata.db."
            )

        cursor.execute(
            "PRAGMA table_info(github_repositories)"
        )

        columns = [
            row[1]
            for row in cursor.fetchall()
        ]

        id_column = None

        for candidate in (
            "repository_id",
            "repo_id",
            "id",
        ):
            if candidate in columns:
                id_column = candidate
                break

        url_column = None

        for candidate in (
            "repository_url",
            "repo_url",
            "github_url",
            "url",
        ):
            if candidate in columns:
                url_column = candidate
                break

        if id_column is None:
            raise RuntimeError(
                "Could not find repository ID column.\n"
                f"Available columns: {columns}"
            )

        if url_column is None:
            raise RuntimeError(
                "Could not find repository URL column.\n"
                f"Available columns: {columns}"
            )

        cursor.execute(
            f"""
            SELECT
                {id_column},
                {url_column}
            FROM github_repositories
            """
        )

        requested_key = (
            normalize_github_repo_key(
                github_url
            )
        )

        rows = cursor.fetchall()

        for repository_id, stored_url in rows:

            stored_key = (
                normalize_github_repo_key(
                    stored_url
                )
            )

            if (
                stored_key
                == requested_key
            ):
                return repository_id

        available = []

        for repository_id, stored_url in rows:
            available.append(
                f"{repository_id} -> {stored_url}"
            )

        available_text = (
            "\n".join(
                available
            )
            if available
            else "No repositories are stored."
        )

        raise RuntimeError(
            "Repository was not found in the GitHub "
            "metadata database.\n\n"
            f"Requested URL:\n{github_url}\n\n"
            f"Normalized key:\n{requested_key}\n\n"
            "Repositories currently stored:\n"
            f"{available_text}\n\n"
            "Click Process Repository again."
        )

    finally:
        connection.close()


# ==============================================================
# GITHUB Q&A
# ==============================================================

def github_answer(
    github_url,
    question,
):
    GitHubCitationRAG = import_class(
        [
            (
                "backend.rag.github_citation_rag",
                "GitHubCitationRAG",
            ),
        ]
    )

    repository_id = (
        get_github_repository_id(
            github_url
        )
    )

    rag = GitHubCitationRAG(
        repository_url=github_url,
        repository_id=repository_id,
    )

    return call_compatible(
        rag,
        [
            "answer",
            "ask",
            "query",
            "run",
        ],
        question,
    )


# ==============================================================
# GITHUB ANALYSIS SERVICE
# ==============================================================

def create_github_analysis_service(
    service_class,
    github_url,
    repository_id,
):
    """
    Inspect the actual constructor and pass only parameters
    that it supports.
    """

    values = {
        "repository_url": github_url,
        "github_url": github_url,
        "repo_url": github_url,
        "repository_id": repository_id,
        "repo_id": repository_id,
        "id": repository_id,
    }

    signature = inspect.signature(
        service_class.__init__
    )

    kwargs = {}

    for parameter in (
        signature.parameters.values()
    ):

        if parameter.name == "self":
            continue

        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if (
            parameter.name
            in values
        ):
            kwargs[
                parameter.name
            ] = values[
                parameter.name
            ]

    return service_class(
        **kwargs
    )


def execute_github_analysis(
    service,
    operation,
    github_url,
    repository_id,
):
    method_map = {
        "summary": [
            "repository_summary",
            "summary",
            "generate_summary",
        ],
        "architecture": [
            "architecture_explanation",
            "architecture",
            "explain_architecture",
        ],
        "both": [
            "summary_and_architecture",
            "analyze",
            "run",
        ],
    }

    method_names = method_map.get(
        operation
    )

    if not method_names:
        raise ValueError(
            f"Unsupported GitHub operation: "
            f"{operation}"
        )

    values = {
        "repository_url": github_url,
        "github_url": github_url,
        "repo_url": github_url,
        "repository_id": repository_id,
        "repo_id": repository_id,
        "id": repository_id,
    }

    errors = []

    for method_name in method_names:

        if not hasattr(
            service,
            method_name,
        ):
            continue

        method = getattr(
            service,
            method_name,
        )

        try:

            signature = inspect.signature(
                method
            )

            kwargs = {}

            for parameter in (
                signature.parameters.values()
            ):

                if parameter.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue

                if (
                    parameter.name
                    in values
                ):
                    kwargs[
                        parameter.name
                    ] = values[
                        parameter.name
                    ]

            return method(
                **kwargs
            )

        except Exception as error:

            errors.append(
                f"{method_name}: {error}"
            )

    raise RuntimeError(
        "Could not execute GitHub analysis.\n"
        + "\n".join(errors)
    )


def github_analysis(
    github_url,
    operation,
):
    AnalysisService = import_class(
        [
            (
                "backend.rag.github_analysis_service",
                "GitHubAnalysisService",
            ),
        ]
    )

    repository_id = (
        get_github_repository_id(
            github_url
        )
    )

    service = (
        create_github_analysis_service(
            AnalysisService,
            github_url,
            repository_id,
        )
    )

    return execute_github_analysis(
        service,
        operation,
        github_url,
        repository_id,
    )


# ==============================================================
# SCOPED RETRIEVAL
# ==============================================================

def scoped_retrieval(
    question,
    source_type,
    source=None,
):
    generator = (
        get_multimodal_generator()
    )

    return generator.router.retrieve(
        question,
        source_type=source_type,
        source=source,
    )


# ==============================================================
# SCOPED SUMMARY
# ==============================================================

def generate_scoped_summary(
    source_type,
    source=None,
):
    retrieval = scoped_retrieval(
        "main topic important information summary",
        source_type,
        source,
    )

    results = retrieval.get(
        "results",
        [],
    )

    if not results:
        raise RuntimeError(
            "No relevant information was found "
            "for the selected source."
        )

    if source_type == "youtube":
        results = results[:3]
    else:
        results = results[:5]

    generator = (
        get_multimodal_generator()
    )

    context = generator._build_context(
        results
    )

    prompt = f"""
You are a grounded multimodal RAG
summarization system.

Create a concise summary using ONLY
the evidence below.

Rules:
1. Do not add outside information.
2. Do not invent facts.
3. Include only information supported
   by the evidence.
4. Do not mention information from
   another source.
5. Do not fabricate citations.

EVIDENCE
========================
{context}
========================

SUMMARY:
"""

    summary = generator.llm.generate(
        prompt
    )

    return {
        "summary": summary,
        "sources": (
            generator._build_sources(
                results
            )
        ),
    }


# ==============================================================
# SCOPED QUESTION GENERATION
# ==============================================================

def generate_scoped_questions(
    source_type,
    source=None,
    number_of_questions=5,
):
    retrieval = scoped_retrieval(
        "important concepts facts details content",
        source_type,
        source,
    )

    results = retrieval.get(
        "results",
        [],
    )

    if not results:
        raise RuntimeError(
            "No relevant information was found "
            "for the selected source."
        )

    if source_type == "youtube":
        results = results[:3]
    else:
        results = results[:5]

    generator = (
        get_multimodal_generator()
    )

    context = generator._build_context(
        results
    )

    prompt = f"""
You are a grounded multimodal RAG
question-generation system.

Generate exactly {number_of_questions}
questions.

Every question must be answerable ONLY
from the evidence below.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. Cover different parts of the evidence.
4. Return only the questions.
5. Number them 1 to {number_of_questions}.

EVIDENCE
========================
{context}
========================

QUESTIONS:
"""

    response = (
        generator.llm.generate(
            prompt
        )
    )

    questions = []

    for line in response.splitlines():

        line = line.strip()

        if not line:
            continue

        if (
            len(line) > 2
            and line[0].isdigit()
            and line[1:3] in (
                ". ",
                ") ",
            )
        ):
            line = line[3:].strip()

        if line.startswith("- "):
            line = line[2:].strip()

        if line:
            questions.append(
                line
            )

    return {
        "questions": questions[
            :number_of_questions
        ],
        "sources": (
            generator._build_sources(
                results
            )
        ),
    }


# ==============================================================
# OUTPUT RENDERING
# ==============================================================

def render_current_output():

    output_type = (
        st.session_state.last_output_type
    )

    if output_type == "question":

        st.subheader(
            "Question"
        )

        st.write(
            st.session_state.last_question
        )

        st.subheader(
            "Answer"
        )

        st.markdown(
            '<div class="answer-box">',
            unsafe_allow_html=True,
        )

        st.write(
            st.session_state.last_answer
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        render_citations(
            st.session_state.last_sources
        )

    elif output_type == "summary":

        st.subheader(
            "Summary"
        )

        st.markdown(
            '<div class="answer-box">',
            unsafe_allow_html=True,
        )

        st.write(
            st.session_state.last_summary
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        render_citations(
            st.session_state.last_sources
        )

    elif output_type == "questions":

        st.subheader(
            "Generated Questions"
        )

        for index, question in enumerate(
            st.session_state.last_questions,
            start=1,
        ):
            st.write(
                f"{index}. {question}"
            )

        render_citations(
            st.session_state.last_sources
        )


# ==============================================================
# HEADER
# ==============================================================

st.markdown(
    '<div class="main-title">'
    "🧠 Universal Multimodal RAG System"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Document • Image • YouTube • GitHub • "
    "Grounded Retrieval • Citations"
    "</div>",
    unsafe_allow_html=True,
)


# ==============================================================
# SIDEBAR
# ==============================================================

with st.sidebar:

    st.header(
        "Source"
    )

    source = st.radio(
        "Select source",
        [
            "Document",
            "Image",
            "YouTube link",
            "GitHub repo link",
        ],
        index=0,
    )

    if (
        st.session_state.active_source
        != source
    ):
        reset_processing()

        st.session_state.active_source = (
            source
        )

    st.divider()

    st.header(
        "LLM"
    )

    provider = st.selectbox(
        "Provider",
        [
            "Ollama",
            "OpenRouter",
            "Gemini",
            "GPT",
        ],
    )

    provider_map = {
        "Ollama": "ollama",
        "OpenRouter": "openrouter",
        "Gemini": "gemini",
        "GPT": "gpt",
    }

    provider_key = provider_map[
        provider
    ]

    model_options = {
        "Ollama": [
            "llama3.2:3b",
        ],
        "OpenRouter": [
            os.getenv(
                "OPENROUTER_MODEL",
                "openai/gpt-5",
            ),
        ],
        "Gemini": [
            os.getenv(
                "GEMINI_MODEL",
                "gemini-3.8-flash",
            ),
        ],
        "GPT": [
            os.getenv(
                "OPENAI_MODEL",
                "gpt-5",
            ),
        ],
    }

    selected_model = st.selectbox(
        "Model",
        model_options[
            provider
        ],
    )

    st.session_state.llm_provider = (
        provider_key
    )

    st.session_state.llm_model = (
        selected_model
    )

    st.divider()

    st.caption(
        "Local Ollama works without cloud API keys. "
        "Cloud providers require credentials in .env."
    )


# ==============================================================
# BACKEND IMPORT ERROR
# ==============================================================

if IMPORT_BOOTSTRAP_ERROR is not None:

    st.error(
        "Backend import error: "
        f"{IMPORT_BOOTSTRAP_ERROR}"
    )


# ==============================================================
# WORKSPACE
# ==============================================================

st.header(
    f"{source} Workspace"
)


# ==============================================================
# DOCUMENT WORKSPACE
# ==============================================================

if source == "Document":

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=[
            "pdf",
            "docx",
            "pptx",
            "txt",
            "csv",
        ],
    )

    if uploaded_file:

        current_hash = (
            get_uploaded_file_hash(
                uploaded_file
            )
        )

        if (
            st.session_state.selected_file_hash
            != current_hash
        ):

            st.session_state.selected_file_hash = (
                current_hash
            )

            st.session_state.processed = False
            st.session_state.processed_source = None
            st.session_state.processed_name = None
            st.session_state.processed_path = None
            st.session_state.processed_retrieval_source = None
            st.session_state.processed_result = None
            st.session_state.processed_file_hash = None

            clear_outputs()

        st.info(
            f"Selected: {uploaded_file.name}"
        )

        if st.button(
            "Process Document",
            type="primary",
            use_container_width=True,
        ):

            try:

                with st.spinner(
                    "Processing document..."
                ):

                    file_path = (
                        save_uploaded_file(
                            uploaded_file
                        )
                    )

                    result = (
                        process_document(
                            file_path
                        )
                    )

                st.session_state.processed = True
                st.session_state.processed_source = (
                    "document"
                )
                st.session_state.processed_name = (
                    uploaded_file.name
                )
                st.session_state.processed_path = (
                    str(file_path)
                )
                st.session_state.processed_retrieval_source = (
                    get_relative_source(
                        file_path
                    )
                )
                st.session_state.processed_result = (
                    result
                )
                st.session_state.processed_file_hash = (
                    current_hash
                )

                clear_outputs()

                st.success(
                    "Document processed successfully."
                )

            except Exception as error:

                st.error(
                    f"Document processing failed: "
                    f"{error}"
                )

        is_current_document = (
            st.session_state.processed_source
            == "document"
            and
            st.session_state.processed_file_hash
            == st.session_state.selected_file_hash
        )

        if is_current_document:

            st.divider()

            st.subheader(
                "Document Actions"
            )

            question = st.text_input(
                "Enter your question",
                key="document_question",
            )

            col1, col2, col3 = (
                st.columns(3)
            )

            with col1:
                ask_clicked = st.button(
                    "Ask Question",
                    key="document_ask",
                    type="primary",
                    use_container_width=True,
                )

            with col2:
                summary_clicked = st.button(
                    "Summarize",
                    key="document_summary",
                    use_container_width=True,
                )

            with col3:
                questions_clicked = st.button(
                    "Generate Questions",
                    key="document_questions",
                    use_container_width=True,
                )

            if ask_clicked:

                clear_outputs()

                if not question.strip():

                    st.warning(
                        "Please enter a question."
                    )

                else:

                    try:

                        with st.spinner(
                            "Searching and generating answer..."
                        ):

                            generator = (
                                get_multimodal_generator()
                            )

                            response = (
                                generator.answer(
                                    question,
                                    source_type="document",
                                    source=(
                                        st.session_state
                                        .processed_retrieval_source
                                    ),
                                )
                            )

                        st.session_state.last_output_type = (
                            "question"
                        )

                        st.session_state.last_question = (
                            question
                        )

                        st.session_state.last_answer = (
                            response.get(
                                "answer",
                                "",
                            )
                        )

                        st.session_state.last_sources = (
                            response.get(
                                "sources",
                                [],
                            )
                        )

                    except Exception as error:

                        st.error(
                            f"Question failed: "
                            f"{error}"
                        )

            elif summary_clicked:

                clear_outputs()

                try:

                    with st.spinner(
                        "Generating summary..."
                    ):

                        result = (
                            generate_scoped_summary(
                                source_type="document",
                                source=(
                                    st.session_state
                                    .processed_retrieval_source
                                ),
                            )
                        )

                    st.session_state.last_output_type = (
                        "summary"
                    )

                    st.session_state.last_summary = (
                        result["summary"]
                    )

                    st.session_state.last_sources = (
                        result["sources"]
                    )

                except Exception as error:

                    st.error(
                        f"Summary failed: "
                        f"{error}"
                    )

            elif questions_clicked:

                clear_outputs()

                try:

                    with st.spinner(
                        "Generating questions..."
                    ):

                        result = (
                            generate_scoped_questions(
                                source_type="document",
                                source=(
                                    st.session_state
                                    .processed_retrieval_source
                                ),
                            )
                        )

                    st.session_state.last_output_type = (
                        "questions"
                    )

                    st.session_state.last_questions = (
                        result["questions"]
                    )

                    st.session_state.last_sources = (
                        result["sources"]
                    )

                except Exception as error:

                    st.error(
                        f"Question generation failed: "
                        f"{error}"
                    )

            render_current_output()


# ==============================================================
# IMAGE WORKSPACE
# ==============================================================

elif source == "Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp",
            "bmp",
            "gif",
        ],
    )

    if uploaded_file:

        current_hash = (
            get_uploaded_file_hash(
                uploaded_file
            )
        )

        if (
            st.session_state.selected_file_hash
            != current_hash
        ):

            st.session_state.selected_file_hash = (
                current_hash
            )

            st.session_state.processed = False
            st.session_state.processed_source = None
            st.session_state.processed_name = None
            st.session_state.processed_path = None
            st.session_state.processed_retrieval_source = None
            st.session_state.processed_result = None
            st.session_state.processed_file_hash = None

            clear_outputs()

        st.image(
            uploaded_file,
            caption=uploaded_file.name,
            use_container_width=True,
        )

        if st.button(
            "Process Image",
            type="primary",
            use_container_width=True,
        ):

            try:

                with st.spinner(
                    "Processing image with OCR and vision..."
                ):

                    file_path = (
                        save_uploaded_file(
                            uploaded_file
                        )
                    )

                    pipeline = (
                        ImageIngestionPipeline()
                    )

                    result = (
                        pipeline.process(
                            file_path
                        )
                    )

                    indexer = (
                        ImageIndexer()
                    )

                    indexer.reindex_image(
                        file_path,
                        result["chunks"],
                    )

                st.session_state.processed = True
                st.session_state.processed_source = (
                    "image"
                )
                st.session_state.processed_name = (
                    uploaded_file.name
                )
                st.session_state.processed_path = (
                    str(file_path)
                )
                st.session_state.processed_retrieval_source = (
                    get_relative_source(
                        file_path
                    )
                )
                st.session_state.processed_result = (
                    result
                )
                st.session_state.processed_file_hash = (
                    current_hash
                )

                clear_outputs()

                st.success(
                    "Image processed and indexed successfully."
                )

            except Exception as error:

                st.error(
                    f"Image processing failed: "
                    f"{error}"
                )

        is_current_image = (
            st.session_state.processed_source
            == "image"
            and
            st.session_state.processed_file_hash
            == st.session_state.selected_file_hash
        )

        if is_current_image:

            st.divider()

            st.subheader(
                "Image Actions"
            )

            question = st.text_input(
                "Enter your question",
                key="image_question",
            )

            col1, col2, col3 = (
                st.columns(3)
            )

            with col1:
                ask_clicked = st.button(
                    "Ask Question",
                    key="image_ask",
                    type="primary",
                    use_container_width=True,
                )

            with col2:
                summary_clicked = st.button(
                    "Summarize",
                    key="image_summary",
                    use_container_width=True,
                )

            with col3:
                questions_clicked = st.button(
                    "Generate Questions",
                    key="image_questions",
                    use_container_width=True,
                )

            if ask_clicked:

                clear_outputs()

                if not question.strip():

                    st.warning(
                        "Please enter a question."
                    )

                else:

                    try:

                        with st.spinner(
                            "Searching image knowledge..."
                        ):

                            generator = (
                                get_multimodal_generator()
                            )

                            response = (
                                generator.answer(
                                    question,
                                    source_type="image",
                                    source=(
                                        st.session_state
                                        .processed_retrieval_source
                                    ),
                                )
                            )

                        st.session_state.last_output_type = (
                            "question"
                        )

                        st.session_state.last_question = (
                            question
                        )

                        st.session_state.last_answer = (
                            response.get(
                                "answer",
                                "",
                            )
                        )

                        st.session_state.last_sources = (
                            response.get(
                                "sources",
                                [],
                            )
                        )

                    except Exception as error:

                        st.error(
                            f"Question failed: "
                            f"{error}"
                        )

            elif summary_clicked:

                clear_outputs()

                try:

                    with st.spinner(
                        "Generating image summary..."
                    ):

                        result = (
                            generate_scoped_summary(
                                source_type="image",
                                source=(
                                    st.session_state
                                    .processed_retrieval_source
                                ),
                            )
                        )

                    st.session_state.last_output_type = (
                        "summary"
                    )

                    st.session_state.last_summary = (
                        result["summary"]
                    )

                    st.session_state.last_sources = (
                        result["sources"]
                    )

                except Exception as error:

                    st.error(
                        f"Summary failed: "
                        f"{error}"
                    )

            elif questions_clicked:

                clear_outputs()

                try:

                    with st.spinner(
                        "Generating image questions..."
                    ):

                        result = (
                            generate_scoped_questions(
                                source_type="image",
                                source=(
                                    st.session_state
                                    .processed_retrieval_source
                                ),
                            )
                        )

                    st.session_state.last_output_type = (
                        "questions"
                    )

                    st.session_state.last_questions = (
                        result["questions"]
                    )

                    st.session_state.last_sources = (
                        result["sources"]
                    )

                except Exception as error:

                    st.error(
                        f"Question generation failed: "
                        f"{error}"
                    )

            render_current_output()


# ==============================================================
# YOUTUBE WORKSPACE
# ==============================================================

elif source == "YouTube link":

    youtube_url = st.text_input(
        "YouTube URL",
        placeholder=(
            "https://www.youtube.com/watch?v=..."
        ),
    )

    answer_language = st.selectbox(
        "Answer Language",
        [
            "English",
            "Same as video",
        ],
    )

    if youtube_url:

        if st.button(
            "Process YouTube Video",
            type="primary",
            use_container_width=True,
        ):

            try:

                with st.spinner(
                    "Loading transcript and indexing video..."
                ):

                    result = (
                        process_youtube(
                            youtube_url
                        )
                    )

                st.session_state.processed = True
                st.session_state.processed_source = (
                    "youtube"
                )
                st.session_state.processed_name = (
                    youtube_url
                )
                st.session_state.processed_path = (
                    youtube_url
                )
                st.session_state.processed_retrieval_source = (
                    youtube_url
                )
                st.session_state.processed_result = (
                    result
                )

                clear_outputs()

                st.success(
                    "YouTube video processed successfully."
                )

            except Exception as error:

                st.error(
                    f"YouTube processing failed: "
                    f"{error}"
                )

    if (
        st.session_state.processed_source
        == "youtube"
        and
        st.session_state.processed_name
        == youtube_url
    ):

        st.divider()

        st.subheader(
            "YouTube Actions"
        )

        question = st.text_input(
            "Enter your question",
            key="youtube_question",
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:
            ask_clicked = st.button(
                "Ask Question",
                key="youtube_ask",
                type="primary",
                use_container_width=True,
            )

        with col2:
            summary_clicked = st.button(
                "Summarize",
                key="youtube_summary",
                use_container_width=True,
            )

        with col3:
            questions_clicked = st.button(
                "Generate Questions",
                key="youtube_questions",
                use_container_width=True,
            )

        if ask_clicked:

            clear_outputs()

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                try:

                    language_mode = (
                        "same"
                        if answer_language
                        == "Same as video"
                        else "english"
                    )

                    YouTubeRAG = import_class(
                        [
                            (
                                "backend.rag.youtube_rag_pipeline",
                                "YouTubeRAGPipeline",
                            ),
                        ]
                    )

                    rag = YouTubeRAG()

                    response = (
                        rag.answer(
                            question,
                            answer_language=language_mode,
                        )
                    )

                    st.session_state.last_output_type = (
                        "question"
                    )

                    st.session_state.last_question = (
                        question
                    )

                    st.session_state.last_answer = (
                        response.get(
                            "answer",
                            "",
                        )
                    )

                    st.session_state.last_sources = (
                        response.get(
                            "sources",
                            [],
                        )
                    )

                except Exception as error:

                    st.error(
                        f"YouTube question failed: "
                        f"{error}"
                    )

        elif summary_clicked:

            clear_outputs()

            try:

                with st.spinner(
                    "Generating video summary..."
                ):

                    result = (
                        generate_scoped_summary(
                            source_type="youtube",
                            source=(
                                st.session_state
                                .processed_retrieval_source
                            ),
                        )
                    )

                st.session_state.last_output_type = (
                    "summary"
                )

                st.session_state.last_summary = (
                    result["summary"]
                )

                st.session_state.last_sources = (
                    result["sources"]
                )

            except Exception as error:

                st.error(
                    f"YouTube summary failed: "
                    f"{error}"
                )

        elif questions_clicked:

            clear_outputs()

            try:

                with st.spinner(
                    "Generating video questions..."
                ):

                    result = (
                        generate_scoped_questions(
                            source_type="youtube",
                            source=(
                                st.session_state
                                .processed_retrieval_source
                            ),
                        )
                    )

                st.session_state.last_output_type = (
                    "questions"
                )

                st.session_state.last_questions = (
                    result["questions"]
                )

                st.session_state.last_sources = (
                    result["sources"]
                )

            except Exception as error:

                st.error(
                    f"YouTube question generation failed: "
                    f"{error}"
                )

        render_current_output()


# ==============================================================
# GITHUB WORKSPACE
# ==============================================================

elif source == "GitHub repo link":

    github_url = st.text_input(
        "Public GitHub repository URL",
        placeholder=(
            "https://github.com/username/repository"
        ),
    )

    if github_url:

        if (
            st.session_state.processed_name
            != github_url
        ):

            if (
                st.session_state.processed_source
                == "github"
            ):
                clear_outputs()

        if st.button(
            "Process Repository",
            type="primary",
            use_container_width=True,
        ):

            try:

                with st.spinner(
                    "Cloning, analyzing, parsing and indexing repository..."
                ):

                    result = (
                        process_github(
                            github_url
                        )
                    )

                st.session_state.processed = True
                st.session_state.processed_source = (
                    "github"
                )
                st.session_state.processed_name = (
                    github_url
                )
                st.session_state.processed_path = (
                    github_url
                )
                st.session_state.processed_retrieval_source = (
                    github_url
                )
                st.session_state.processed_result = (
                    result
                )

                clear_outputs()

                st.success(
                    "GitHub repository processed successfully."
                )

            except Exception as error:

                st.error(
                    f"GitHub processing failed: "
                    f"{error}"
                )

    if (
        st.session_state.processed_source
        == "github"
        and
        st.session_state.processed_name
        == github_url
    ):

        st.divider()

        st.subheader(
            "Search / Ask Question"
        )

        question = st.text_input(
            "Enter your repository question",
            key="github_question",
            placeholder=(
                "How many files are there in the repository?"
            ),
        )

        if st.button(
            "Ask Question",
            key="github_ask",
            type="primary",
            use_container_width=True,
        ):

            clear_outputs()

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                try:

                    with st.spinner(
                        "Searching repository..."
                    ):

                        response = (
                            github_answer(
                                github_url,
                                question,
                            )
                        )

                    st.session_state.last_output_type = (
                        "question"
                    )

                    st.session_state.last_question = (
                        question
                    )

                    if isinstance(
                        response,
                        dict,
                    ):

                        st.session_state.last_answer = (
                            response.get(
                                "answer",
                                "",
                            )
                        )

                        st.session_state.last_sources = (
                            response.get(
                                "sources",
                                [],
                            )
                        )

                    else:

                        st.session_state.last_answer = (
                            str(response)
                        )

                except Exception as error:

                    st.error(
                        f"GitHub question failed: "
                        f"{error}"
                    )

        render_current_output()

        st.divider()

        st.subheader(
            "Repository Analysis"
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:
            summary_clicked = st.button(
                "Repository Summary",
                key="github_summary",
                use_container_width=True,
            )

        with col2:
            architecture_clicked = st.button(
                "Architecture Explanation",
                key="github_architecture",
                use_container_width=True,
            )

        with col3:
            both_clicked = st.button(
                "Summary + Architecture",
                key="github_both",
                use_container_width=True,
            )

        operation = None

        if summary_clicked:
            operation = "summary"

        elif architecture_clicked:
            operation = "architecture"

        elif both_clicked:
            operation = "both"

        if operation:

            try:

                with st.spinner(
                    "Analyzing repository..."
                ):

                    result = (
                        github_analysis(
                            github_url,
                            operation,
                        )
                    )

                if isinstance(
                    result,
                    dict,
                ):

                    if operation == "summary":

                        st.subheader(
                            "Repository Summary"
                        )

                        st.write(
                            result.get(
                                "summary",
                                result,
                            )
                        )

                    elif operation == "architecture":

                        st.subheader(
                            "Architecture Explanation"
                        )

                        st.write(
                            result.get(
                                "architecture",
                                result,
                            )
                        )

                    else:

                        if result.get(
                            "summary"
                        ):

                            st.subheader(
                                "Repository Summary"
                            )

                            st.write(
                                result[
                                    "summary"
                                ]
                            )

                        if result.get(
                            "architecture"
                        ):

                            st.subheader(
                                "Architecture Explanation"
                            )

                            st.write(
                                result[
                                    "architecture"
                                ]
                            )

                else:

                    st.write(
                        result
                    )

            except Exception as error:

                st.error(
                    f"GitHub analysis failed: "
                    f"{error}"
                )


# ==============================================================
# FOOTER
# ==============================================================

st.divider()

st.caption(
    "Universal Multimodal RAG System • "
    "Document + Image + YouTube + GitHub • "
    "Grounded retrieval with source citations"
)