import ast
from pathlib import Path


class GitHubCodeParser:

    def __init__(self):
        self.supported_extensions = {
            ".py": "Python"
        }

    def is_supported_file(self, file_path):
        extension = Path(file_path).suffix.lower()

        return extension in self.supported_extensions

    def read_file(self, file_path):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File does not exist: {file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                f"Path is not a file: {file_path}"
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

    def extract_imports(self, tree):

        imports = []

        for node in ast.walk(tree):

            if isinstance(node, ast.Import):

                for alias in node.names:
                    imports.append({
                        "name": alias.name,
                        "alias": alias.asname,
                        "type": "import",
                        "line": node.lineno
                    })

            elif isinstance(node, ast.ImportFrom):

                module = node.module or ""

                for alias in node.names:
                    imports.append({
                        "name": (
                            f"{module}.{alias.name}"
                            if module
                            else alias.name
                        ),
                        "alias": alias.asname,
                        "type": "from_import",
                        "line": node.lineno
                    })

        return imports

    def get_end_line(self, node):

        end_line = getattr(
            node,
            "end_lineno",
            None
        )

        if end_line is not None:
            return end_line

        return node.lineno

    def extract_functions(
        self,
        tree,
        parent_class=None
    ):

        functions = []

        for node in ast.walk(tree):

            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef
                )
            ):

                functions.append({
                    "name": node.name,
                    "type": (
                        "async_function"
                        if isinstance(
                            node,
                            ast.AsyncFunctionDef
                        )
                        else "function"
                    ),
                    "parent_class": parent_class,
                    "start_line": node.lineno,
                    "end_line": self.get_end_line(node),
                    "arguments": [
                        argument.arg
                        for argument in node.args.args
                    ],
                    "decorators": [
                        self._get_node_name(
                            decorator
                        )
                        for decorator in node.decorator_list
                    ]
                })

        return functions

    def extract_classes(self, tree):

        classes = []

        for node in ast.walk(tree):

            if not isinstance(
                node,
                ast.ClassDef
            ):
                continue

            methods = []

            for child in node.body:

                if isinstance(
                    child,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef
                    )
                ):

                    methods.append({
                        "name": child.name,
                        "type": (
                            "async_method"
                            if isinstance(
                                child,
                                ast.AsyncFunctionDef
                            )
                            else "method"
                        ),
                        "start_line": child.lineno,
                        "end_line": self.get_end_line(
                            child
                        ),
                        "arguments": [
                            argument.arg
                            for argument in child.args.args
                        ],
                        "decorators": [
                            self._get_node_name(
                                decorator
                            )
                            for decorator
                            in child.decorator_list
                        ]
                    })

            classes.append({
                "name": node.name,
                "start_line": node.lineno,
                "end_line": self.get_end_line(node),
                "bases": [
                    self._get_node_name(base)
                    for base in node.bases
                ],
                "methods": methods
            })

        return classes

    def extract_top_level_functions(
        self,
        tree
    ):

        functions = []

        for node in tree.body:

            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef
                )
            ):

                functions.append({
                    "name": node.name,
                    "type": (
                        "async_function"
                        if isinstance(
                            node,
                            ast.AsyncFunctionDef
                        )
                        else "function"
                    ),
                    "parent_class": None,
                    "start_line": node.lineno,
                    "end_line": self.get_end_line(node),
                    "arguments": [
                        argument.arg
                        for argument in node.args.args
                    ],
                    "decorators": [
                        self._get_node_name(
                            decorator
                        )
                        for decorator
                        in node.decorator_list
                    ]
                })

        return functions

    def _get_node_name(self, node):

        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.Attribute):

            parts = []

            current = node

            while isinstance(
                current,
                ast.Attribute
            ):
                parts.append(current.attr)
                current = current.value

            if isinstance(
                current,
                ast.Name
            ):
                parts.append(current.id)

            return ".".join(
                reversed(parts)
            )

        if isinstance(node, ast.Call):
            return self._get_node_name(
                node.func
            )

        return ast.dump(node)

    def extract_function_calls(self, tree):

        calls = []

        for node in ast.walk(tree):

            if not isinstance(
                node,
                ast.Call
            ):
                continue

            function_name = self._get_node_name(
                node.func
            )

            calls.append({
                "name": function_name,
                "line": node.lineno
            })

        return calls

    def parse_python_file(
        self,
        file_path,
        relative_path=None
    ):

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
                "path": (
                    str(relative_path)
                    if relative_path
                    else str(file_path)
                ),
                "language": "Python",
                "parse_status": "failed",
                "error": str(error),
                "source_code": source_code,
                "imports": [],
                "classes": [],
                "functions": [],
                "function_calls": [],
                "line_count": len(
                    source_code.splitlines()
                )
            }

        classes = self.extract_classes(
            tree
        )

        functions = self.extract_top_level_functions(
            tree
        )

        imports = self.extract_imports(
            tree
        )

        function_calls = self.extract_function_calls(
            tree
        )

        path = (
            str(relative_path)
            if relative_path
            else str(file_path)
        )

        return {
            "path": path,
            "language": "Python",
            "parse_status": "success",
            "error": None,
            "source_code": source_code,
            "imports": imports,
            "classes": classes,
            "functions": functions,
            "function_calls": function_calls,
            "line_count": len(
                source_code.splitlines()
            )
        }

    def parse_file(
        self,
        file_path,
        relative_path=None
    ):

        file_path = Path(file_path)

        if not self.is_supported_file(
            file_path
        ):

            return {
                "path": (
                    str(relative_path)
                    if relative_path
                    else str(file_path)
                ),
                "language": "Unknown",
                "parse_status": "unsupported",
                "error": (
                    "File extension is not "
                    "supported by the code parser."
                ),
                "source_code": None,
                "imports": [],
                "classes": [],
                "functions": [],
                "function_calls": [],
                "line_count": 0
            }

        return self.parse_python_file(
            file_path,
            relative_path
        )

    def parse_repository(
        self,
        repository_path
    ):

        repository_path = Path(
            repository_path
        )

        if not repository_path.exists():
            raise FileNotFoundError(
                f"Repository path does not exist: "
                f"{repository_path}"
            )

        parsed_files = []

        for file_path in repository_path.rglob(
            "*"
        ):

            if not file_path.is_file():
                continue

            relative_path = file_path.relative_to(
                repository_path
            )

            if not self.is_supported_file(
                file_path
            ):
                continue

            parsed_file = self.parse_file(
                file_path,
                relative_path
            )

            parsed_files.append(
                parsed_file
            )

        parsed_files.sort(
            key=lambda item: item["path"].lower()
        )

        return parsed_files