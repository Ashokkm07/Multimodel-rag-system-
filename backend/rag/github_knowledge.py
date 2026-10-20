from pathlib import Path

from backend.processing.github_analyzer import (
    GitHubRepositoryAnalyzer
)
from backend.processing.github_code_parser import (
    GitHubCodeParser
)
from backend.processing.github_code_chunker import (
    GitHubCodeChunker
)


class GitHubKnowledgeBuilder:

    def __init__(self):
        self.analyzer = GitHubRepositoryAnalyzer()
        self.parser = GitHubCodeParser()
        self.chunker = GitHubCodeChunker()

    def build(self, repository_path):

        repository_path = Path(
            repository_path
        )

        if not repository_path.exists():
            raise FileNotFoundError(
                f"Repository does not exist: "
                f"{repository_path}"
            )

        analysis = (
            self.analyzer.analyze_repository(
                repository_path
            )
        )

        parsed_files = (
            self.parser.parse_repository(
                repository_path
            )
        )

        chunks = (
            self.chunker.chunk_repository(
                repository_path
            )
        )

        return {
            "repository_path": str(repository_path),
            "analysis": analysis,
            "parsed_files": parsed_files,
            "chunks": chunks
        }