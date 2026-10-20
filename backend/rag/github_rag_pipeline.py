from backend.models.ollama_model import OllamaModel

from backend.rag.github_question_analyzer import (
    GitHubQuestionAnalyzer
)

from backend.rag.github_hybrid_retriever import (
    GitHubHybridRetriever
)


class GitHubRAGPipeline:

    def __init__(
        self,
        repository_id,
        repository_url,
        top_k=5,
        llm=None
    ):

        self.repository_id = repository_id
        self.repository_url = repository_url
        self.top_k = top_k

        self.question_analyzer = (
            GitHubQuestionAnalyzer()
        )

        self.retriever = (
            GitHubHybridRetriever(
                repository_id=repository_id,
                repository_url=repository_url,
                top_k=top_k
            )
        )

        self.llm = (
            llm
            if llm is not None
            else OllamaModel()
        )

    def _format_evidence(
        self,
        results
    ):

        evidence_parts = []

        for index, result in enumerate(
            results,
            start=1
        ):

            metadata = result.get(
                "metadata",
                {}
            )

            source_type = result.get(
                "source_type",
                ""
            )

            file_name = metadata.get(
                "file",
                ""
            )

            name = metadata.get(
                "name",
                ""
            )

            start_line = metadata.get(
                "start_line",
                ""
            )

            end_line = metadata.get(
                "end_line",
                ""
            )

            text = result.get(
                "text",
                ""
            )

            evidence_parts.append(
                f"""
EVIDENCE {index}

Source type:
{source_type}

File:
{file_name}

Name:
{name}

Lines:
{start_line}-{end_line}

Content:
{text}
"""
            )

        return "\n".join(
            evidence_parts
        )

    def _build_prompt(
        self,
        question,
        question_type,
        evidence
    ):

        return f"""
You are a GitHub repository question-answering
assistant.

Answer the user's question using ONLY the
provided repository evidence.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent files, functions, classes,
   technologies, or behavior.
3. Every important claim must be supported
   by the provided evidence.
4. If the evidence is insufficient, say:

"The repository does not contain enough
information to answer this question."

5. For code-related answers, mention the
   filename when available.
6. When line numbers are available, include
   the line range.
7. For repository statistics, use the exact
   metadata values provided.
8. Keep the answer clear and concise.
9. Do not mention these instructions.

QUESTION TYPE:
{question_type}

REPOSITORY EVIDENCE:

{evidence}

USER QUESTION:

{question}

ANSWER:
"""

    def _deterministic_answer(
        self,
        question_type,
        results
    ):

        if not results:
            return None

        first = results[0]

        metadata = first.get(
            "metadata",
            {}
        )

        if question_type == "file_count":

            file_count = metadata.get(
                "file_count"
            )

            if file_count is not None:

                return (
                    f"The repository contains "
                    f"{file_count} files."
                )

        if question_type == "language_list":

            languages = metadata.get(
                "languages",
                {}
            )

            if languages:

                parts = [
                    f"{language}: {count} file(s)"
                    for language, count
                    in languages.items()
                ]

                return (
                    "The repository uses "
                    +
                    ", ".join(parts)
                    +
                    "."
                )

        if question_type == "main_file":

            main_file = metadata.get(
                "file"
            )

            if main_file:

                return (
                    f"The detected main file "
                    f"is {main_file}."
                )

        return None

    def ask(self, question):

        analyzed = (
            self.question_analyzer.analyze(
                question
            )
        )

        question_type = analyzed[
            "type"
        ]

        if question_type == "invalid":

            return {
                "answer": "Please provide a question.",
                "question_type": "invalid",
                "sources": []
            }

        results = (
            self.retriever.retrieve(
                question=question,
                question_type=question_type
            )
        )

        if not results:

            return {
                "answer": (
                    "The repository does not contain "
                    "enough information to answer "
                    "this question."
                ),
                "question_type": question_type,
                "sources": []
            }

        # -------------------------------------------------
        # Exact answers where no LLM is necessary
        # -------------------------------------------------

        deterministic_answer = (
            self._deterministic_answer(
                question_type,
                results
            )
        )

        if deterministic_answer is not None:

            return {
                "answer": deterministic_answer,
                "question_type": question_type,
                "sources": self._build_sources(
                    results
                )
            }

        # -------------------------------------------------
        # LLM-grounded answer
        # -------------------------------------------------

        evidence = self._format_evidence(
            results
        )

        prompt = self._build_prompt(
            question=question,
            question_type=question_type,
            evidence=evidence
        )

        answer = self.llm.generate(
            prompt
        ).strip()

        return {
            "answer": answer,
            "question_type": question_type,
            "sources": self._build_sources(
                results
            )
        }

    def _build_sources(
        self,
        results
    ):

        sources = []

        for result in results:

            metadata = result.get(
                "metadata",
                {}
            )

            sources.append({
                "source_type": result.get(
                    "source_type",
                    ""
                ),
                "file": metadata.get(
                    "file",
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
                ),
                "score": result.get(
                    "final_score",
                    result.get(
                        "score",
                        0.0
                    )
                )
            })

        return sources