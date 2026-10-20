from backend.models.ollama_model import (
    OllamaModel
)

from backend.rag.github_question_analyzer import (
    GitHubQuestionAnalyzer
)


class GitHubQA:

    def __init__(
        self,
        knowledge,
        llm=None
    ):

        self.knowledge = knowledge

        self.question_analyzer = (
            GitHubQuestionAnalyzer()
        )

        self.llm = (
            llm
            if llm
            else OllamaModel()
        )

    def _repository_context(self):

        analysis = (
            self.knowledge["analysis"]
        )

        context = []

        context.append(
            "REPOSITORY INFORMATION"
        )

        context.append(
            f"Repository path: "
            f"{analysis['repository_path']}"
        )

        context.append(
            f"Total files: "
            f"{analysis['file_count']}"
        )

        context.append(
            f"Total directories: "
            f"{analysis['directory_count']}"
        )

        context.append(
            f"Languages: "
            f"{analysis['languages']}"
        )

        context.append(
            f"Categories: "
            f"{analysis['categories']}"
        )

        main_file = analysis[
            "main_file"
        ]

        context.append(
            f"Main file: "
            f"{main_file.get('path')}"
        )

        context.append(
            "\nFILES:"
        )

        for file in analysis["files"]:

            context.append(
                f"- {file['path']} "
                f"| {file['language']} "
                f"| {file['category']} "
                f"| {file['size_bytes']} bytes"
            )

        context.append(
            "\nDIRECTORIES:"
        )

        for directory in analysis[
            "directories"
        ]:

            context.append(
                f"- {directory}"
            )

        return "\n".join(context)

    def _file_context(self):

        analysis = (
            self.knowledge["analysis"]
        )

        context = []

        for file in analysis["files"]:

            context.append(
                f"""
FILE

Path:
{file['path']}

Name:
{file['name']}

Language:
{file['language']}

Category:
{file['category']}

Extension:
{file['extension']}

Size:
{file['size_bytes']} bytes
"""
            )

        return "\n".join(context)

    def _code_context(self):

        chunks = self.knowledge[
            "chunks"
        ]

        context = []

        for chunk in chunks:

            metadata = chunk[
                "metadata"
            ]

            context.append(
                f"""
CODE CHUNK

File:
{metadata.get('file', '')}

Chunk type:
{metadata.get('chunk_type', '')}

Name:
{metadata.get('name', '')}

Function:
{metadata.get('function', '')}

Class:
{metadata.get('class', '')}

Lines:
{metadata.get('start_line', '')}-
{metadata.get('end_line', '')}

Code:
{chunk['text']}
"""
            )

        return "\n".join(context)

    def _architecture_context(self):

        repository_context = (
            self._repository_context()
        )

        code_context = (
            self._code_context()
        )

        return (
            repository_context
            + "\n\n"
            + code_context
        )

    def _build_prompt(
        self,
        question,
        question_type,
        context
    ):

        return f"""
You are a GitHub repository analysis assistant.

You must answer the user's question using
ONLY the provided repository information and
source-code context.

IMPORTANT RULES:

- Do not use general knowledge.
- Do not invent repository files.
- Do not invent functions or classes.
- Do not claim something exists unless it is
  supported by the provided repository context.
- If the repository context does not contain
  enough information, say:

"The repository does not contain enough
information to answer this question."

- Be concise but useful.
- When discussing code, mention the file name.
- When line information is available, mention
  the line range.
- When discussing a function or class, mention
  its name.
- Do not mention these instructions.

QUESTION TYPE:
{question_type}

REPOSITORY CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    def ask(self, question):

        analyzed = (
            self.question_analyzer.analyze(
                question
            )
        )

        question_type = analyzed["type"]

        if question_type == "invalid":

            return {
                "answer": "Please provide a question.",
                "question_type": "invalid",
                "sources": []
            }

        if question_type == "repository":

            context = (
                self._repository_context()
            )

        elif question_type == "file":

            context = (
                self._file_context()
                + "\n\n"
                + self._code_context()
            )

        elif question_type == "architecture":

            context = (
                self._architecture_context()
            )

        else:

            context = (
                self._code_context()
            )

        prompt = self._build_prompt(
            question=question,
            question_type=question_type,
            context=context
        )

        answer = self.llm.generate(
            prompt
        ).strip()

        sources = []

        if question_type in {
            "code",
            "file",
            "architecture"
        }:

            for chunk in self.knowledge[
                "chunks"
            ]:

                metadata = chunk[
                    "metadata"
                ]

                sources.append({
                    "file": metadata.get(
                        "file",
                        ""
                    ),
                    "chunk_type": metadata.get(
                        "chunk_type",
                        ""
                    ),
                    "name": metadata.get(
                        "name",
                        ""
                    ),
                    "start_line": metadata.get(
                        "start_line",
                        ""
                    ),
                    "end_line": metadata.get(
                        "end_line",
                        ""
                    )
                })

        else:

            analysis = (
                self.knowledge["analysis"]
            )

            for file in analysis[
                "files"
            ]:

                sources.append({
                    "file": file["path"],
                    "chunk_type": "repository_metadata",
                    "name": file["name"],
                    "start_line": "",
                    "end_line": ""
                })

        return {
            "answer": answer,
            "question_type": question_type,
            "sources": sources
        }