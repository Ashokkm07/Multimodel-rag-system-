from pathlib import Path


class GitHubRepositoryAnalyzer:

    def __init__(self):
        """
        Initialize the GitHub repository analyzer.
        """

        # Files/directories that should not be analyzed.
        self.ignored_directories = {
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            "node_modules",
            ".idea",
            ".vscode"
        }

        # Common source-code extensions.
        self.language_extensions = {
            ".py": "Python",
            ".js": "JavaScript",
            ".jsx": "JavaScript React",
            ".ts": "TypeScript",
            ".tsx": "TypeScript React",
            ".java": "Java",
            ".c": "C",
            ".cpp": "C++",
            ".h": "C/C++ Header",
            ".hpp": "C++ Header",
            ".cs": "C#",
            ".go": "Go",
            ".rs": "Rust",
            ".php": "PHP",
            ".rb": "Ruby",
            ".swift": "Swift",
            ".kt": "Kotlin",
            ".kts": "Kotlin",
            ".scala": "Scala",
            ".r": "R",
            ".sql": "SQL",
            ".sh": "Shell",
            ".bash": "Shell",
            ".ps1": "PowerShell",
            ".html": "HTML",
            ".css": "CSS",
            ".scss": "SCSS",
            ".vue": "Vue",
            ".dart": "Dart"
        }

    def _is_ignored(self, path):
        """
        Check whether a file or directory belongs to
        an ignored directory.
        """

        for part in path.parts:

            if part in self.ignored_directories:
                return True

        return False

    def discover_files(self, repository_path):
        """
        Discover all files inside the repository.

        Git metadata and common virtual/environment
        directories are ignored.
        """

        repository_path = Path(
            repository_path
        )

        if not repository_path.exists():

            raise FileNotFoundError(
                f"Repository path does not exist: "
                f"{repository_path}"
            )

        if not repository_path.is_dir():

            raise ValueError(
                f"Repository path is not a directory: "
                f"{repository_path}"
            )

        files = []

        for path in repository_path.rglob("*"):

            if not path.is_file():
                continue

            relative_path = path.relative_to(
                repository_path
            )

            if self._is_ignored(
                relative_path
            ):
                continue

            files.append(
                relative_path
            )

        files.sort(
            key=lambda path: str(path).lower()
        )

        return files

    def discover_directories(
        self,
        repository_path
    ):
        """
        Discover all directories inside the repository.
        """

        repository_path = Path(
            repository_path
        )

        directories = []

        for path in repository_path.rglob("*"):

            if not path.is_dir():
                continue

            relative_path = path.relative_to(
                repository_path
            )

            if self._is_ignored(
                relative_path
            ):
                continue

            directories.append(
                relative_path
            )

        directories.sort(
            key=lambda path: str(path).lower()
        )

        return directories

    def get_file_extension(
        self,
        file_path
    ):
        """
        Return the lowercase file extension.
        """

        return Path(
            file_path
        ).suffix.lower()

    def detect_language(
        self,
        file_path
    ):
        """
        Detect the programming language from
        the file extension.
        """

        extension = self.get_file_extension(
            file_path
        )

        return self.language_extensions.get(
            extension,
            "Unknown"
        )

    def classify_file(
        self,
        file_path
    ):
        """
        Classify a repository file.

        Categories include:

        - source_code
        - documentation
        - configuration
        - dependency
        - model
        - dataset
        - notebook
        - image
        - other
        """

        path = Path(
            file_path
        )

        name = path.name.lower()

        extension = path.suffix.lower()

        # -------------------------------------------------
        # Documentation
        # -------------------------------------------------

        if name in {
            "readme",
            "readme.md",
            "readme.txt",
            "license",
            "license.md",
            "license.txt"
        }:

            return "documentation"

        # -------------------------------------------------
        # Source code
        # -------------------------------------------------

        if extension in self.language_extensions:

            return "source_code"

        # -------------------------------------------------
        # Configuration
        # -------------------------------------------------

        if extension in {
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
            ".env"
        }:

            return "configuration"

        # -------------------------------------------------
        # Dependency files
        # -------------------------------------------------

        if name in {
            "requirements.txt",
            "package.json",
            "package-lock.json",
            "yarn.lock",
            "poetry.lock",
            "pipfile",
            "pipfile.lock"
        }:

            return "dependency"

        # -------------------------------------------------
        # Machine learning models
        # -------------------------------------------------

        if extension in {
            ".pkl",
            ".pickle",
            ".joblib",
            ".pt",
            ".pth",
            ".onnx",
            ".h5",
            ".keras",
            ".safetensors"
        }:

            return "model"

        # -------------------------------------------------
        # Datasets
        # -------------------------------------------------

        if extension in {
            ".csv",
            ".tsv",
            ".jsonl",
            ".parquet",
            ".xlsx",
            ".xls"
        }:

            return "dataset"

        # -------------------------------------------------
        # Jupyter notebooks
        # -------------------------------------------------

        if extension == ".ipynb":

            return "notebook"

        # -------------------------------------------------
        # Images
        # -------------------------------------------------

        if extension in {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".webp",
            ".svg"
        }:

            return "image"

        return "other"

    def get_file_size(
        self,
        repository_path,
        relative_file_path
    ):
        """
        Return the file size in bytes.
        """

        repository_path = Path(
            repository_path
        )

        file_path = (
            repository_path /
            relative_file_path
        )

        return file_path.stat().st_size

    def analyze_files(
        self,
        repository_path
    ):
        """
        Analyze all files in a repository.
        """

        repository_path = Path(
            repository_path
        )

        files = self.discover_files(
            repository_path
        )

        file_information = []

        for relative_path in files:

            extension = (
                self.get_file_extension(
                    relative_path
                )
            )

            language = (
                self.detect_language(
                    relative_path
                )
            )

            category = (
                self.classify_file(
                    relative_path
                )
            )

            size = self.get_file_size(
                repository_path,
                relative_path
            )

            file_information.append({
                "path": str(
                    relative_path
                ),
                "name": relative_path.name,
                "extension": extension,
                "language": language,
                "category": category,
                "size_bytes": size
            })

        return file_information

    def identify_main_file(
        self,
        file_information
    ):
        """
        Identify the most likely main/entry file.

        This is heuristic-based. It does not claim that
        the selected file is definitely the application's
        entry point.
        """

        priority_names = [
            "main.py",
            "app.py",
            "application.py",
            "server.py",
            "index.py",
            "index.js",
            "index.ts",
            "main.js",
            "main.ts",
            "manage.py",
            "run.py",
            "server.js",
            "server.ts"
        ]

        # -------------------------------------------------
        # First: exact filename priority
        # -------------------------------------------------

        for priority_name in priority_names:

            for file in file_information:

                if (
                    file["name"].lower()
                    == priority_name
                ):

                    return {
                        "path": file["path"],
                        "name": file["name"],
                        "reason": (
                            "The filename matches a "
                            "common application entry-point "
                            "name."
                        ),
                        "confidence": "high"
                    }

        # -------------------------------------------------
        # Second: Python files
        # -------------------------------------------------

        python_files = [
            file
            for file in file_information
            if file["language"] == "Python"
        ]

        if len(python_files) == 1:

            return {
                "path": python_files[0]["path"],
                "name": python_files[0]["name"],
                "reason": (
                    "This is the only Python source "
                    "file in the repository."
                ),
                "confidence": "medium"
            }

        # -------------------------------------------------
        # Third: JavaScript / TypeScript
        # -------------------------------------------------

        js_files = [
            file
            for file in file_information
            if file["language"]
            in {
                "JavaScript",
                "JavaScript React",
                "TypeScript",
                "TypeScript React"
            }
        ]

        if len(js_files) == 1:

            return {
                "path": js_files[0]["path"],
                "name": js_files[0]["name"],
                "reason": (
                    "This is the only JavaScript/"
                    "TypeScript source file."
                ),
                "confidence": "medium"
            }

        # -------------------------------------------------
        # No clear main file
        # -------------------------------------------------

        return {
            "path": None,
            "name": None,
            "reason": (
                "No clear application entry point "
                "could be identified from the "
                "repository structure."
            ),
            "confidence": "low"
        }

    def analyze_repository(
        self,
        repository_path
    ):
        """
        Perform complete repository-level analysis.
        """

        repository_path = Path(
            repository_path
        )

        file_information = (
            self.analyze_files(
                repository_path
            )
        )

        directories = (
            self.discover_directories(
                repository_path
            )
        )

        # -------------------------------------------------
        # Language statistics
        # -------------------------------------------------

        languages = {}

        for file in file_information:

            language = file[
                "language"
            ]

            if language == "Unknown":
                continue

            languages[language] = (
                languages.get(
                    language,
                    0
                ) + 1
            )

        # -------------------------------------------------
        # Category statistics
        # -------------------------------------------------

        categories = {}

        for file in file_information:

            category = file[
                "category"
            ]

            categories[category] = (
                categories.get(
                    category,
                    0
                ) + 1
            )

        # -------------------------------------------------
        # Main file
        # -------------------------------------------------

        main_file = (
            self.identify_main_file(
                file_information
            )
        )

        # -------------------------------------------------
        # Repository summary
        # -------------------------------------------------

        return {
            "repository_path": str(
                repository_path
            ),
            "file_count": len(
                file_information
            ),
            "directory_count": len(
                directories
            ),
            "files": file_information,
            "directories": [
                str(directory)
                for directory in directories
            ],
            "languages": languages,
            "categories": categories,
            "main_file": main_file
        }