from pathlib import Path

from backend.ingestion.github_loader import GitHubLoader
from backend.processing.github_analyzer import (
    GitHubRepositoryAnalyzer
)
from backend.processing.github_code_parser import (
    GitHubCodeParser
)
from backend.processing.github_code_chunker import (
    GitHubCodeChunker
)


class GitHubRepositoryPipeline:

    def __init__(self):
        self.loader = GitHubLoader()
        self.analyzer = GitHubRepositoryAnalyzer()
        self.parser = GitHubCodeParser()
        self.chunker = GitHubCodeChunker()

    def process(self, github_url):

        print("\n" + "=" * 70)
        print("             GITHUB REPOSITORY PIPELINE")
        print("=" * 70)

        # -------------------------------------------------
        # 1. Validate and extract repository information
        # -------------------------------------------------

        print("\n[1] REPOSITORY INFORMATION")
        print("-" * 70)

        self.loader.validate_url(github_url)

        repository_info = (
            self.loader.extract_repository_info(
                github_url
            )
        )

        print(
            f"Owner      : "
            f"{repository_info['owner']}"
        )

        print(
            f"Repository : "
            f"{repository_info['repository']}"
        )

        print(
            f"URL        : "
            f"{repository_info['url']}"
        )

        # -------------------------------------------------
        # 2. Clone repository
        # -------------------------------------------------

        print("\n[2] CLONING REPOSITORY")
        print("-" * 70)

        clone_result = self.loader.clone_repository(
            github_url
        )

        repository_path = Path(
            clone_result["local_path"]
        )

        print(
            f"Status     : "
            f"{clone_result['status']}"
        )

        print(
            f"Local path : "
            f"{repository_path}"
        )

        # -------------------------------------------------
        # 3. Repository analysis
        # -------------------------------------------------

        print("\n[3] REPOSITORY ANALYSIS")
        print("-" * 70)

        analysis = self.analyzer.analyze_repository(
            repository_path
        )

        print(
            f"Total files       : "
            f"{analysis['file_count']}"
        )

        print(
            f"Total directories : "
            f"{analysis['directory_count']}"
        )

        print("\nLanguages:")

        for language, count in (
            analysis["languages"].items()
        ):
            print(
                f"  {language}: {count}"
            )

        print("\nCategories:")

        for category, count in (
            analysis["categories"].items()
        ):
            print(
                f"  {category}: {count}"
            )

        main_file = analysis["main_file"]

        print("\nMain file:")

        if main_file["path"]:

            print(
                f"  Path       : "
                f"{main_file['path']}"
            )

            print(
                f"  Confidence : "
                f"{main_file['confidence']}"
            )

            print(
                f"  Reason     : "
                f"{main_file['reason']}"
            )

        else:

            print(
                "  No clear main file identified."
            )

        # -------------------------------------------------
        # 4. Parse source files
        # -------------------------------------------------

        print("\n[4] CODE PARSING")
        print("-" * 70)

        parsed_files = self.parser.parse_repository(
            repository_path
        )

        print(
            f"Python files parsed : "
            f"{len(parsed_files)}"
        )

        total_functions = 0
        total_classes = 0

        for parsed_file in parsed_files:

            total_functions += len(
                parsed_file["functions"]
            )

            total_classes += len(
                parsed_file["classes"]
            )

        print(
            f"Classes found       : "
            f"{total_classes}"
        )

        print(
            f"Functions found     : "
            f"{total_functions}"
        )

        # -------------------------------------------------
        # 5. Intelligent code chunking
        # -------------------------------------------------

        print("\n[5] INTELLIGENT CODE CHUNKING")
        print("-" * 70)

        chunks = self.chunker.chunk_repository(
            repository_path
        )

        print(
            f"Total code chunks : "
            f"{len(chunks)}"
        )

        chunk_types = {}

        for chunk in chunks:

            chunk_type = chunk[
                "metadata"
            ]["chunk_type"]

            chunk_types[chunk_type] = (
                chunk_types.get(
                    chunk_type,
                    0
                ) + 1
            )

        print("\nChunk types:")

        for chunk_type, count in (
            chunk_types.items()
        ):

            print(
                f"  {chunk_type}: {count}"
            )

        # -------------------------------------------------
        # 6. Return complete result
        # -------------------------------------------------

        return {
            "github_url": repository_info["url"],
            "owner": repository_info["owner"],
            "repository": repository_info["repository"],
            "local_path": str(repository_path),
            "clone_status": clone_result["status"],
            "analysis": analysis,
            "parsed_files": parsed_files,
            "chunks": chunks
        }