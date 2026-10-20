from backend.utils.github_database import (
    GitHubDatabase
)


class GitHubMetadataService:

    def __init__(self):

        self.database = GitHubDatabase()

    def save_repository(
        self,
        pipeline_result
    ):

        repository_info = {
            "owner": pipeline_result["owner"],
            "repository": pipeline_result[
                "repository"
            ],
            "url": pipeline_result[
                "github_url"
            ],
            "local_path": pipeline_result[
                "local_path"
            ]
        }

        analysis = pipeline_result[
            "analysis"
        ]

        existing = (
            self.database.repository_exists(
                repository_info["url"]
            )
        )

        if existing["exists"]:

            return {
                "status": "already_exists",
                "repository_id": existing["id"]
            }

        repository_id = (
            self.database.add_repository(
                owner=repository_info["owner"],
                repository=repository_info[
                    "repository"
                ],
                url=repository_info["url"],
                local_path=repository_info[
                    "local_path"
                ],
                file_count=analysis[
                    "file_count"
                ],
                directory_count=analysis[
                    "directory_count"
                ],
                main_file=analysis[
                    "main_file"
                ].get("path")
            )
        )

        self.database.add_files(
            repository_id,
            analysis["files"]
        )

        self.database.add_functions(
            repository_id,
            pipeline_result[
                "parsed_files"
            ]
        )

        self.database.add_classes(
            repository_id,
            pipeline_result[
                "parsed_files"
            ]
        )

        return {
            "status": "saved",
            "repository_id": repository_id
        }

    def get_repository_summary(
        self,
        repository_id
    ):

        repository = (
            self.database.get_repository(
                repository_id
            )
        )

        if not repository:
            return None

        files = (
            self.database.get_files(
                repository_id
            )
        )

        functions = (
            self.database.get_functions(
                repository_id
            )
        )

        classes = (
            self.database.get_classes(
                repository_id
            )
        )

        return {
            "repository": repository,
            "files": files,
            "functions": functions,
            "classes": classes
        }