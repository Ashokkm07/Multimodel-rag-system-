import sqlite3
from pathlib import Path


DATABASE_PATH = "data/rag_metadata.db"


class GitHubDatabase:

    def __init__(
        self,
        database_path=DATABASE_PATH
    ):
        self.database_path = database_path

        Path(
            database_path
        ).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.create_tables()

    def get_connection(self):
        return sqlite3.connect(
            self.database_path
        )

    def create_tables(self):

        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS github_repositories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner TEXT NOT NULL,
                repository TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                local_path TEXT NOT NULL,
                file_count INTEGER DEFAULT 0,
                directory_count INTEGER DEFAULT 0,
                main_file TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS github_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repository_id INTEGER NOT NULL,
                path TEXT NOT NULL,
                name TEXT NOT NULL,
                extension TEXT,
                language TEXT,
                category TEXT,
                size_bytes INTEGER DEFAULT 0,
                FOREIGN KEY(repository_id)
                    REFERENCES github_repositories(id),
                UNIQUE(repository_id, path)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS github_functions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repository_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                function_name TEXT NOT NULL,
                function_type TEXT,
                parent_class TEXT,
                start_line INTEGER,
                end_line INTEGER,
                FOREIGN KEY(repository_id)
                    REFERENCES github_repositories(id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS github_classes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repository_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                class_name TEXT NOT NULL,
                start_line INTEGER,
                end_line INTEGER,
                FOREIGN KEY(repository_id)
                    REFERENCES github_repositories(id)
            )
            """
        )

        connection.commit()
        connection.close()

    def repository_exists(self, url):

        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                owner,
                repository,
                local_path
            FROM github_repositories
            WHERE url = ?
            """,
            (url,)
        )

        result = cursor.fetchone()

        connection.close()

        if result:

            return {
                "exists": True,
                "id": result[0],
                "owner": result[1],
                "repository": result[2],
                "local_path": result[3]
            }

        return {
            "exists": False
        }

    def add_repository(
        self,
        owner,
        repository,
        url,
        local_path,
        file_count,
        directory_count,
        main_file
    ):

        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO github_repositories (
                owner,
                repository,
                url,
                local_path,
                file_count,
                directory_count,
                main_file
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                owner,
                repository,
                url,
                local_path,
                file_count,
                directory_count,
                main_file
            )
        )

        connection.commit()

        repository_id = cursor.lastrowid

        connection.close()

        return repository_id

    def add_files(
        self,
        repository_id,
        files
    ):

        connection = self.get_connection()
        cursor = connection.cursor()

        for file in files:

            cursor.execute(
                """
                INSERT OR REPLACE INTO github_files (
                    repository_id,
                    path,
                    name,
                    extension,
                    language,
                    category,
                    size_bytes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    repository_id,
                    file["path"],
                    file["name"],
                    file["extension"],
                    file["language"],
                    file["category"],
                    file["size_bytes"]
                )
            )

        connection.commit()
        connection.close()

    def add_functions(
        self,
        repository_id,
        parsed_files
    ):

        connection = self.get_connection()
        cursor = connection.cursor()

        for parsed_file in parsed_files:

            file_path = parsed_file["path"]

            for function in parsed_file[
                "functions"
            ]:

                cursor.execute(
                    """
                    INSERT INTO github_functions (
                        repository_id,
                        file_path,
                        function_name,
                        function_type,
                        parent_class,
                        start_line,
                        end_line
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        repository_id,
                        file_path,
                        function["name"],
                        function["type"],
                        function.get(
                            "parent_class"
                        ),
                        function["start_line"],
                        function["end_line"]
                    )
                )

            for class_info in parsed_file[
                "classes"
            ]:

                for method in class_info[
                    "methods"
                ]:

                    cursor.execute(
                        """
                        INSERT INTO github_functions (
                            repository_id,
                            file_path,
                            function_name,
                            function_type,
                            parent_class,
                            start_line,
                            end_line
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            repository_id,
                            file_path,
                            method["name"],
                            method["type"],
                            class_info["name"],
                            method["start_line"],
                            method["end_line"]
                        )
                    )

        connection.commit()
        connection.close()

    def add_classes(
        self,
        repository_id,
        parsed_files
    ):

        connection = self.get_connection()
        cursor = connection.cursor()

        for parsed_file in parsed_files:

            file_path = parsed_file["path"]

            for class_info in parsed_file[
                "classes"
            ]:

                cursor.execute(
                    """
                    INSERT INTO github_classes (
                        repository_id,
                        file_path,
                        class_name,
                        start_line,
                        end_line
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        repository_id,
                        file_path,
                        class_info["name"],
                        class_info["start_line"],
                        class_info["end_line"]
                    )
                )

        connection.commit()
        connection.close()

    def get_repository(
        self,
        repository_id
    ):

        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                owner,
                repository,
                url,
                local_path,
                file_count,
                directory_count,
                main_file
            FROM github_repositories
            WHERE id = ?
            """,
            (repository_id,)
        )

        result = cursor.fetchone()

        connection.close()

        if not result:
            return None

        return {
            "id": result[0],
            "owner": result[1],
            "repository": result[2],
            "url": result[3],
            "local_path": result[4],
            "file_count": result[5],
            "directory_count": result[6],
            "main_file": result[7]
        }

    def get_files(
        self,
        repository_id
    ):

        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                path,
                name,
                extension,
                language,
                category,
                size_bytes
            FROM github_files
            WHERE repository_id = ?
            ORDER BY path
            """,
            (repository_id,)
        )

        rows = cursor.fetchall()

        connection.close()

        return [
            {
                "path": row[0],
                "name": row[1],
                "extension": row[2],
                "language": row[3],
                "category": row[4],
                "size_bytes": row[5]
            }
            for row in rows
        ]

    def get_functions(
        self,
        repository_id
    ):

        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                file_path,
                function_name,
                function_type,
                parent_class,
                start_line,
                end_line
            FROM github_functions
            WHERE repository_id = ?
            ORDER BY file_path, start_line
            """,
            (repository_id,)
        )

        rows = cursor.fetchall()

        connection.close()

        return [
            {
                "file": row[0],
                "name": row[1],
                "type": row[2],
                "parent_class": row[3],
                "start_line": row[4],
                "end_line": row[5]
            }
            for row in rows
        ]

    def get_classes(
        self,
        repository_id
    ):

        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                file_path,
                class_name,
                start_line,
                end_line
            FROM github_classes
            WHERE repository_id = ?
            ORDER BY file_path, start_line
            """,
            (repository_id,)
        )

        rows = cursor.fetchall()

        connection.close()

        return [
            {
                "file": row[0],
                "name": row[1],
                "start_line": row[2],
                "end_line": row[3]
            }
            for row in rows
        ]