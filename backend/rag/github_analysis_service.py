from backend.models.ollama_model import OllamaModel


class GitHubAnalysisService:

    def __init__(
        self,
        knowledge,
        llm=None
    ):
        self.knowledge = knowledge

        self.llm = (
            llm
            if llm is not None
            else OllamaModel()
        )

    # =========================================================
    # REPOSITORY CONTEXT
    # =========================================================

    def _repository_context(self):

        analysis = self.knowledge["analysis"]

        parts = []

        parts.append(
            "REPOSITORY INFORMATION"
        )

        parts.append(
            f"Repository path: "
            f"{analysis['repository_path']}"
        )

        parts.append(
            f"Total files: "
            f"{analysis['file_count']}"
        )

        parts.append(
            f"Total directories: "
            f"{analysis['directory_count']}"
        )

        parts.append(
            f"Languages: "
            f"{analysis['languages']}"
        )

        parts.append(
            f"Categories: "
            f"{analysis['categories']}"
        )

        main_file = analysis.get(
            "main_file",
            {}
        )

        parts.append(
            f"Main file: "
            f"{main_file.get('path')}"
        )

        parts.append(
            f"Main file confidence: "
            f"{main_file.get('confidence')}"
        )

        parts.append(
            "\nFILES:"
        )

        for file in analysis["files"]:

            parts.append(
                f"- {file['path']} | "
                f"Language: {file['language']} | "
                f"Category: {file['category']} | "
                f"Size: {file['size_bytes']} bytes"
            )

        parts.append(
            "\nDIRECTORIES:"
        )

        for directory in analysis[
            "directories"
        ]:

            parts.append(
                f"- {directory}"
            )

        return "\n".join(parts)

    # =========================================================
    # CODE CONTEXT
    # =========================================================

    def _code_context(self):

        parts = []

        for chunk in self.knowledge[
            "chunks"
        ]:

            metadata = chunk[
                "metadata"
            ]

            parts.append(
                f"""
FILE:
{metadata.get('file', '')}

CHUNK TYPE:
{metadata.get('chunk_type', '')}

NAME:
{metadata.get('name', '')}

FUNCTION:
{metadata.get('function', '')}

CLASS:
{metadata.get('class', '')}

LINES:
{metadata.get('start_line', '')}-
{metadata.get('end_line', '')}

CODE:
{chunk['text']}
"""
            )

        return "\n".join(parts)

    # =========================================================
    # SUMMARY
    # =========================================================

    def summarize(self):

        repository_context = (
            self._repository_context()
        )

        prompt = f"""
You are a GitHub repository analysis assistant.

Create a repository summary using ONLY the
provided repository information.

Include:

1. Repository purpose, only when supported
   by the available evidence.
2. Main application or entry-point file.
3. Programming languages.
4. Important files.
5. Important file categories.
6. High-level functionality supported
   by the repository evidence.

IMPORTANT RULES:

- Do not use outside knowledge.
- Do not invent functionality.
- Do not invent files.
- Do not assume what an undocumented
  file contains.
- Clearly state when the available evidence
  is insufficient.
- Keep the summary concise.
- Mention relevant filenames.

REPOSITORY INFORMATION:

{repository_context}

SUMMARY:
"""

        return self.llm.generate(
            prompt
        ).strip()

    # =========================================================
    # ARCHITECTURE
    # =========================================================

    def explain_architecture(self):

        repository_context = (
            self._repository_context()
        )

        code_context = (
            self._code_context()
        )

        prompt = f"""
You are a GitHub repository architecture
analysis assistant.

Explain the repository architecture using
ONLY the provided repository information
and source code.

Cover:

1. Main entry point.
2. Major files and their apparent roles.
3. Main processing flow.
4. Model, data, configuration, or dependency
   relationships when supported.
5. How the application appears to operate.
6. Important limitations or unknown parts.

IMPORTANT RULES:

- Do not use outside knowledge.
- Do not invent components.
- Do not invent relationships.
- Do not claim functionality that is not
  supported by the evidence.
- Distinguish clearly between observed
  implementation and uncertain interpretation.
- Mention filenames and line ranges when available.
- Keep the explanation structured and readable.

REPOSITORY INFORMATION:

{repository_context}

SOURCE CODE:

{code_context}

ARCHITECTURE:
"""

        return self.llm.generate(
            prompt
        ).strip()