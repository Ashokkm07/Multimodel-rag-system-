import re

from backend.embeddings.github_indexer import (
    GitHubIndexer
)

from backend.utils.github_database import (
    GitHubDatabase
)


class GitHubHybridRetriever:

    def __init__(
        self,
        repository_id,
        repository_url,
        top_k=5
    ):

        self.repository_id = repository_id
        self.repository_url = repository_url
        self.top_k = top_k

        self.database = GitHubDatabase()
        self.indexer = GitHubIndexer()

    # =========================================================
    # TEXT UTILITIES
    # =========================================================

    def _tokenize(self, text):

        text = text.lower()

        return set(
            re.findall(
                r"[a-zA-Z0-9_]+",
                text
            )
        )

    def _lexical_score(
        self,
        question,
        text
    ):

        question_tokens = self._tokenize(
            question
        )

        text_tokens = self._tokenize(
            text
        )

        if not question_tokens:
            return 0.0

        overlap = (
            question_tokens
            & text_tokens
        )

        return (
            len(overlap)
            /
            len(question_tokens)
        )

    # =========================================================
    # SQLITE METADATA
    # =========================================================

    def get_repository_metadata(self):

        repository = (
            self.database.get_repository(
                self.repository_id
            )
        )

        files = (
            self.database.get_files(
                self.repository_id
            )
        )

        functions = (
            self.database.get_functions(
                self.repository_id
            )
        )

        classes = (
            self.database.get_classes(
                self.repository_id
            )
        )

        return {
            "repository": repository,
            "files": files,
            "functions": functions,
            "classes": classes
        }

    # =========================================================
    # EXACT REPOSITORY RETRIEVAL
    # =========================================================

    def retrieve_file_count(self):

        metadata = (
            self.get_repository_metadata()
        )

        repository = metadata["repository"]

        return [{
            "source_type": "repository_metadata",
            "text": (
                f"Repository contains "
                f"{repository['file_count']} files."
            ),
            "metadata": {
                "type": "file_count",
                "file_count": repository[
                    "file_count"
                ]
            },
            "score": 1.0
        }]

    def retrieve_file_list(self):

        metadata = (
            self.get_repository_metadata()
        )

        candidates = []

        for file in metadata["files"]:

            candidates.append({
                "source_type": "metadata_file",
                "text": (
                    f"File: {file['path']}"
                ),
                "metadata": {
                    "type": "file",
                    **file
                },
                "score": 1.0
            })

        return candidates

    def retrieve_languages(self):

        metadata = (
            self.get_repository_metadata()
        )

        language_counts = {}

        for file in metadata["files"]:

            language = file["language"]

            if language == "Unknown":
                continue

            language_counts[language] = (
                language_counts.get(
                    language,
                    0
                ) + 1
            )

        result_text = []

        for language, count in sorted(
            language_counts.items()
        ):

            result_text.append(
                f"{language}: {count} file(s)"
            )

        return [{
            "source_type": "repository_metadata",
            "text": (
                "Programming languages used: "
                +
                ", ".join(result_text)
            ),
            "metadata": {
                "type": "language_list",
                "languages": language_counts
            },
            "score": 1.0
        }]

    def retrieve_main_file(self):

        metadata = (
            self.get_repository_metadata()
        )

        repository = metadata[
            "repository"
        ]

        main_file = repository[
            "main_file"
        ]

        return [{
            "source_type": "repository_metadata",
            "text": (
                f"Main file: "
                f"{main_file}"
            ),
            "metadata": {
                "type": "main_file",
                "file": main_file
            },
            "score": 1.0
        }]

    def retrieve_functions(self):

        metadata = (
            self.get_repository_metadata()
        )

        candidates = []

        for function in metadata[
            "functions"
        ]:

            candidates.append({
                "source_type": "metadata_function",
                "text": (
                    f"Function "
                    f"{function['name']} "
                    f"in "
                    f"{function['file']}"
                ),
                "metadata": {
                    "type": "function",
                    **function
                },
                "score": 1.0
            })

        return candidates

    def retrieve_classes(self):

        metadata = (
            self.get_repository_metadata()
        )

        candidates = []

        for class_info in metadata[
            "classes"
        ]:

            candidates.append({
                "source_type": "metadata_class",
                "text": (
                    f"Class "
                    f"{class_info['name']} "
                    f"in "
                    f"{class_info['file']}"
                ),
                "metadata": {
                    "type": "class",
                    **class_info
                },
                "score": 1.0
            })

        return candidates

    # =========================================================
    # FILE / CODE SEARCH
    # =========================================================

    def retrieve_file_content(
        self,
        question
    ):

        metadata = (
            self.get_repository_metadata()
        )

        candidates = []

        for file in metadata["files"]:

            searchable_text = " ".join([
                file["path"],
                file["name"],
                file["extension"],
                file["language"],
                file["category"]
            ])

            score = self._lexical_score(
                question,
                searchable_text
            )

            if score > 0:

                candidates.append({
                    "source_type": "metadata_file",
                    "text": searchable_text,
                    "metadata": {
                        "type": "file",
                        **file
                    },
                    "score": score
                })

        return candidates

    def retrieve_semantic(
        self,
        question,
        top_k=None
    ):

        if top_k is None:
            top_k = self.top_k * 3

        results = self.indexer.search(
            repository_url=self.repository_url,
            query=question,
            top_k=top_k
        )

        candidates = []

        for result in results:

            metadata = result[
                "metadata"
            ]

            lexical_score = (
                self._lexical_score(
                    question,
                    result["text"]
                )
            )

            candidates.append({
                "source_type": "semantic",
                "text": result["text"],
                "metadata": metadata,
                "semantic_distance": result[
                    "distance"
                ],
                "lexical_score": lexical_score
            })

        return candidates

    # =========================================================
    # RERANK SEMANTIC RESULTS
    # =========================================================

    def rerank_code(
        self,
        question,
        candidates
    ):

        if not candidates:
            return []

        distances = [
            candidate["semantic_distance"]
            for candidate in candidates
            if candidate.get(
                "semantic_distance"
            ) is not None
        ]

        min_distance = (
            min(distances)
            if distances
            else 0.0
        )

        max_distance = (
            max(distances)
            if distances
            else 0.0
        )

        for candidate in candidates:

            distance = candidate.get(
                "semantic_distance"
            )

            if distance is None:

                semantic_score = 0.0

            elif max_distance == min_distance:

                semantic_score = 1.0

            else:

                semantic_score = (
                    1.0
                    -
                    (
                        distance
                        - min_distance
                    )
                    /
                    (
                        max_distance
                        - min_distance
                    )
                )

            lexical_score = candidate.get(
                "lexical_score",
                0.0
            )

            candidate[
                "semantic_score"
            ] = semantic_score

            candidate[
                "final_score"
            ] = (
                0.70 * semantic_score
                +
                0.30 * lexical_score
            )

        candidates.sort(
            key=lambda item: item[
                "final_score"
            ],
            reverse=True
        )

        return candidates[
            :self.top_k
        ]

    # =========================================================
    # MAIN ROUTER
    # =========================================================

    def retrieve(
        self,
        question,
        question_type="code"
    ):

        # -----------------------------------------------------
        # Exact deterministic questions
        # -----------------------------------------------------

        if question_type == "file_count":

            return self.retrieve_file_count()

        if question_type == "file_list":

            return self.retrieve_file_list()

        if question_type == "language_list":

            return self.retrieve_languages()

        if question_type == "main_file":

            return self.retrieve_main_file()

        if question_type == "function_list":

            return self.retrieve_functions()

        if question_type == "class_list":

            return self.retrieve_classes()

        # -----------------------------------------------------
        # File question
        # -----------------------------------------------------

        if question_type == "file_content":

            metadata_results = (
                self.retrieve_file_content(
                    question
                )
            )

            semantic_results = (
                self.retrieve_semantic(
                    question
                )
            )

            semantic_results = (
                self.rerank_code(
                    question,
                    semantic_results
                )
            )

            return (
                metadata_results
                +
                semantic_results
            )[:self.top_k]

        # -----------------------------------------------------
        # Architecture
        # -----------------------------------------------------

        if question_type == "architecture":

            metadata = (
                self.get_repository_metadata()
            )

            results = [{
                "source_type": "repository_metadata",
                "text": (
                    f"Repository: "
                    f"{metadata['repository']}"
                ),
                "metadata": {
                    "type": "repository"
                },
                "final_score": 1.0
            }]

            semantic_results = (
                self.retrieve_semantic(
                    question
                )
            )

            semantic_results = (
                self.rerank_code(
                    question,
                    semantic_results
                )
            )

            results.extend(
                semantic_results
            )

            return results[
                :self.top_k
            ]

        # -----------------------------------------------------
        # Code questions
        # -----------------------------------------------------

        semantic_results = (
            self.retrieve_semantic(
                question
            )
        )

        return self.rerank_code(
            question,
            semantic_results
        )