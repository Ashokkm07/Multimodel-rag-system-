import re
import shutil
import subprocess
from pathlib import Path


class GitHubLoader:

    def __init__(
        self,
        download_directory="data/uploads/github"
    ):
        """
        Initialize the GitHub repository loader.

        Repositories will be cloned into:

        data/uploads/github/
        """

        self.download_directory = Path(
            download_directory
        )

        self.download_directory.mkdir(
            parents=True,
            exist_ok=True
        )

    def validate_url(self, github_url):
        """
        Validate a GitHub repository URL.
        """

        if not github_url:
            raise ValueError(
                "GitHub URL cannot be empty."
            )

        github_url = github_url.strip()

        pattern = (
            r"^https?://github\.com/"
            r"[^/\s]+/"
            r"[^/\s]+"
            r"/?$"
        )

        if not re.match(
            pattern,
            github_url
        ):
            raise ValueError(
                "Invalid GitHub repository URL. "
                "Expected format: "
                "https://github.com/owner/repository"
            )

        return True

    def extract_repository_info(
        self,
        github_url
    ):
        """
        Extract the GitHub owner and repository name.
        """

        github_url = github_url.strip()

        github_url = github_url.rstrip(
            "/"
        )

        match = re.match(
            r"^https?://github\.com/"
            r"([^/]+)/([^/]+)$",
            github_url
        )

        if not match:

            raise ValueError(
                "Could not extract GitHub repository "
                "information from the URL."
            )

        owner = match.group(1)

        repository = match.group(2)

        if repository.endswith(
            ".git"
        ):

            repository = repository[
                :-4
            ]

        return {
            "owner": owner,
            "repository": repository,
            "url": (
                f"https://github.com/"
                f"{owner}/"
                f"{repository}"
            )
        }

    def clone_repository(
        self,
        github_url
    ):
        """
        Clone a GitHub repository locally.

        Returns repository information and
        the local repository path.
        """

        # -------------------------------------------------
        # 1. Validate URL
        # -------------------------------------------------

        self.validate_url(
            github_url
        )

        # -------------------------------------------------
        # 2. Extract repository information
        # -------------------------------------------------

        repository_info = (
            self.extract_repository_info(
                github_url
            )
        )

        owner = repository_info[
            "owner"
        ]

        repository = repository_info[
            "repository"
        ]

        # -------------------------------------------------
        # 3. Create owner directory
        # -------------------------------------------------

        owner_directory = (
            self.download_directory /
            owner
        )

        owner_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        local_repository_path = (
            owner_directory /
            repository
        )

        # -------------------------------------------------
        # 4. Handle existing repository
        # -------------------------------------------------

        if local_repository_path.exists():

            print(
                "\nRepository already exists locally."
            )

            print(
                f"Local path: "
                f"{local_repository_path}"
            )

            return {
                **repository_info,
                "local_path": str(
                    local_repository_path
                ),
                "status": "already_exists"
            }

        # -------------------------------------------------
        # 5. Clone repository
        # -------------------------------------------------

        print(
            "\nCloning GitHub repository..."
        )

        print(
            f"Repository: "
            f"{repository_info['url']}"
        )

        print(
            f"Destination: "
            f"{local_repository_path}"
        )

        try:

            result = subprocess.run(
                [
                    "git",
                    "clone",
                    repository_info["url"],
                    str(
                        local_repository_path
                    )
                ],
                capture_output=True,
                text=True,
                check=True
            )

        except FileNotFoundError:

            raise RuntimeError(
                "Git was not found on this system. "
                "Please install Git and make sure "
                "it is available in PATH."
            )

        except subprocess.CalledProcessError as error:

            # Remove incomplete clone if created
            if local_repository_path.exists():

                shutil.rmtree(
                    local_repository_path
                )

            error_message = (
                error.stderr.strip()
                if error.stderr
                else "Unknown Git error."
            )

            raise RuntimeError(
                "GitHub repository cloning failed: "
                f"{error_message}"
            )

        # -------------------------------------------------
        # 6. Return result
        # -------------------------------------------------

        print(
            "\nRepository cloned successfully."
        )

        return {
            **repository_info,
            "local_path": str(
                local_repository_path
            ),
            "status": "cloned"
        }