from backend.rag.github_rag_pipeline import (
    GitHubRAGPipeline
)

from backend.rag.github_citation_formatter import (
    GitHubCitationFormatter
)


class GitHubCitationRAG:

    def __init__(
        self,
        repository_id,
        repository_url,
        top_k=5,
        llm=None
    ):

        self.rag = GitHubRAGPipeline(
            repository_id=repository_id,
            repository_url=repository_url,
            top_k=top_k,
            llm=llm
        )

    def ask(self, question):

        result = self.rag.ask(
            question
        )

        citations = (
            GitHubCitationFormatter
            .format_sources(
                result.get(
                    "sources",
                    []
                )
            )
        )

        return {
            "answer": result[
                "answer"
            ],
            "question_type": result[
                "question_type"
            ],
            "citations": citations
        }