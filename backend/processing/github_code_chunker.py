import ast
from pathlib import Path


class GitHubCodeChunker:

    def __init__(self):
        self.supported_extensions = {
            ".py": "Python"
        }

    def read_file(self, file_path):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File does not exist: {file_path}"
            )

        try:
            return file_path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:
            return file_path.read_text(
                encoding="utf-8",
                errors="replace"
            )

    def get_line_offsets(self, source_code):
        """
        Create character offsets for every line.

        This allows AST line numbers to be converted
        into exact source-code sections.
        """

        lines = source_code.splitlines(
            keepends=True
        )

        offsets = [0]

        current_offset = 0

        for line in lines:
            current_offset += len(line)
            offsets.append(current_offset)

        return lines, offsets

    def extract_node_source(
        self,
        source_code,
        node,
        lines,
        offsets
    ):
        """
        Extract exact source code belonging to an AST node.
        """

        start_line = node.lineno
        end_line = getattr(
            node,
            "end_lineno",
            start_line
        )

        start_index = offsets[
            start_line - 1
        ]

        end_index = offsets[
            end_line
        ]

        return source_code[
            start_index:end_index
        ].rstrip()

    def create_chunk(
        self,
        source_code,
        node,
        lines,
        offsets,
        relative_path,
        chunk_type,
        name=None,
        parent_class=None
    ):
        start_line = node.lineno

        end_line = getattr(
            node,
            "end_lineno",
            start_line
        )

        code = self.extract_node_source(
            source_code,
            node,
            lines,
            offsets
        )

        return {
            "text": code,
            "metadata": {
                "file": str(relative_path),
                "language": "Python",
                "chunk_type": chunk_type,
                "name": name or "",
                "function": (
                    name
                    if chunk_type == "function"
                    else ""
                ),
                "class": (
                    parent_class
                    if parent_class
                    else (
                        name
                        if chunk_type == "class"
                        else ""
                    )
                ),
                "start_line": start_line,
                "end_line": end_line
            }
        }

    def extract_import_chunk(
        self,
        source_code,
        tree,
        relative_path
    ):
        """
        Group all top-level imports into one chunk.
        """

        import_nodes = []

        for node in tree.body:

            if isinstance(
                node,
                (
                    ast.Import,
                    ast.ImportFrom
                )
            ):
                import_nodes.append(node)

        if not import_nodes:
            return None

        first_node = import_nodes[0]
        last_node = import_nodes[-1]

        start_line = first_node.lineno

        end_line = getattr(
            last_node,
            "end_lineno",
            last_node.lineno
        )

        lines = source_code.splitlines(
            keepends=True
        )

        code = "".join(
            lines[
                start_line - 1:end_line
            ]
        ).rstrip()

        return {
            "text": code,
            "metadata": {
                "file": str(relative_path),
                "language": "Python",
                "chunk_type": "imports",
                "name": "imports",
                "function": "",
                "class": "",
                "start_line": start_line,
                "end_line": end_line
            }
        }

    def extract_global_code_chunk(
        self,
        source_code,
        tree,
        relative_path
    ):
        """
        Extract top-level executable statements
        that are not imports, functions, or classes.

        This captures application setup such as:

        model loading
        configuration
        variables
        Streamlit setup
        etc.
        """

        global_nodes = []

        for node in tree.body:

            if isinstance(
                node,
                (
                    ast.Import,
                    ast.ImportFrom,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef
                )
            ):
                continue

            global_nodes.append(node)

        if not global_nodes:
            return None

        first_node = global_nodes[0]
        last_node = global_nodes[-1]

        start_line = first_node.lineno

        end_line = getattr(
            last_node,
            "end_lineno",
            last_node.lineno
        )

        lines = source_code.splitlines(
            keepends=True
        )

        code = "".join(
            lines[
                start_line - 1:end_line
            ]
        ).rstrip()

        return {
            "text": code,
            "metadata": {
                "file": str(relative_path),
                "language": "Python",
                "chunk_type": "global_code",
                "name": "global_code",
                "function": "",
                "class": "",
                "start_line": start_line,
                "end_line": end_line
            }
        }

    def extract_functions(
        self,
        source_code,
        tree,
        relative_path
    ):
        """
        Extract top-level functions as individual chunks.
        """

        lines, offsets = self.get_line_offsets(
            source_code
        )

        chunks = []

        for node in tree.body:

            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef
                )
            ):
                continue

            chunk = self.create_chunk(
                source_code=source_code,
                node=node,
                lines=lines,
                offsets=offsets,
                relative_path=relative_path,
                chunk_type="function",
                name=node.name
            )

            chunks.append(chunk)

        return chunks

    def extract_classes(
        self,
        source_code,
        tree,
        relative_path
    ):
        """
        Extract classes as chunks.

        Each class is also represented as one complete
        chunk so the model can understand the class
        as a whole.
        """

        lines, offsets = self.get_line_offsets(
            source_code
        )

        chunks = []

        for node in tree.body:

            if not isinstance(
                node,
                ast.ClassDef
            ):
                continue

            chunk = self.create_chunk(
                source_code=source_code,
                node=node,
                lines=lines,
                offsets=offsets,
                relative_path=relative_path,
                chunk_type="class",
                name=node.name
            )

            chunks.append(chunk)

        return chunks

    def extract_methods(
        self,
        source_code,
        tree,
        relative_path
    ):
        """
        Extract individual class methods.

        This gives the RAG system finer-grained
        retrieval than class-level chunks alone.
        """

        lines, offsets = self.get_line_offsets(
            source_code
        )

        chunks = []

        for class_node in tree.body:

            if not isinstance(
                class_node,
                ast.ClassDef
            ):
                continue

            for node in class_node.body:

                if not isinstance(
                    node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef
                    )
                ):
                    continue

                chunk = self.create_chunk(
                    source_code=source_code,
                    node=node,
                    lines=lines,
                    offsets=offsets,
                    relative_path=relative_path,
                    chunk_type="method",
                    name=node.name,
                    parent_class=class_node.name
                )

                chunks.append(chunk)

        return chunks

    def parse_and_chunk(
        self,
        file_path,
        relative_path=None
    ):
        """
        Parse one Python file and create
        structure-aware chunks.
        """

        file_path = Path(file_path)

        source_code = self.read_file(
            file_path
        )

        try:
            tree = ast.parse(
                source_code
            )

        except SyntaxError as error:

            return {
                "file": (
                    str(relative_path)
                    if relative_path
                    else str(file_path)
                ),
                "language": "Python",
                "status": "failed",
                "error": str(error),
                "chunks": []
            }

        if relative_path is None:
            relative_path = file_path.name

        chunks = []

        import_chunk = (
            self.extract_import_chunk(
                source_code,
                tree,
                relative_path
            )
        )

        if import_chunk:
            chunks.append(import_chunk)

        global_chunk = (
            self.extract_global_code_chunk(
                source_code,
                tree,
                relative_path
            )
        )

        if global_chunk:
            chunks.append(global_chunk)

        chunks.extend(
            self.extract_classes(
                source_code,
                tree,
                relative_path
            )
        )

        chunks.extend(
            self.extract_methods(
                source_code,
                tree,
                relative_path
            )
        )

        chunks.extend(
            self.extract_functions(
                source_code,
                tree,
                relative_path
            )
        )

        chunks.sort(
            key=lambda chunk: (
                chunk["metadata"]["start_line"],
                chunk["metadata"]["end_line"],
                chunk["metadata"]["chunk_type"]
            )
        )

        for index, chunk in enumerate(
            chunks,
            start=1
        ):
            chunk["metadata"]["chunk_id"] = (
                index
            )

        return {
            "file": str(relative_path),
            "language": "Python",
            "status": "success",
            "error": None,
            "chunks": chunks
        }

    def chunk_repository(
        self,
        repository_path
    ):
        """
        Chunk all supported source files
        in a repository.
        """

        repository_path = Path(
            repository_path
        )

        if not repository_path.exists():
            raise FileNotFoundError(
                f"Repository path does not exist: "
                f"{repository_path}"
            )

        all_chunks = []

        for file_path in repository_path.rglob(
            "*"
        ):

            if not file_path.is_file():
                continue

            extension = (
                file_path.suffix.lower()
            )

            if extension not in (
                self.supported_extensions
            ):
                continue

            relative_path = file_path.relative_to(
                repository_path
            )

            result = self.parse_and_chunk(
                file_path,
                relative_path
            )

            all_chunks.extend(
                result["chunks"]
            )

        return all_chunks