
import sqlite3
from pathlib import Path


DATABASE_PATH = "data/rag_metadata.db"


class Database:

    def __init__(self, database_path=DATABASE_PATH):
        """
        Initialize the SQLite database.
        """

        Path(database_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.database_path = database_path

        self.create_tables()

    def get_connection(self):
        """
        Create and return a connection to the SQLite database.
        """

        return sqlite3.connect(
            self.database_path
        )

    def create_tables(self):
        """
        Create the documents table if it does not already exist.
        """

        connection = self.get_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                filename TEXT NOT NULL,

                file_path TEXT NOT NULL,

                file_hash TEXT NOT NULL UNIQUE,

                file_type TEXT NOT NULL,

                processing_status TEXT NOT NULL,

                chunk_count INTEGER DEFAULT 0,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.commit()

        connection.close()

    def add_document(
        self,
        filename,
        file_path,
        file_hash,
        file_type,
        processing_status="processing",
        chunk_count=0
    ):
        """
        Register a new document in the database.
        """

        connection = self.get_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO documents (
                filename,
                file_path,
                file_hash,
                file_type,
                processing_status,
                chunk_count
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                file_path,
                file_hash,
                file_type,
                processing_status,
                chunk_count
            )
        )

        connection.commit()

        document_id = cursor.lastrowid

        connection.close()

        return document_id

    def is_duplicate(self, file_hash):
        """
        Check whether a file with the given SHA-256 hash
        already exists in the database.

        Returns information about the existing document
        if the hash is found.
        """

        connection = self.get_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                filename,
                processing_status
            FROM documents
            WHERE file_hash = ?
            """,
            (file_hash,)
        )

        result = cursor.fetchone()

        connection.close()

        if result:

            return {
                "exists": True,
                "id": result[0],
                "filename": result[1],
                "processing_status": result[2]
            }

        return {
            "exists": False
        }

    def update_processing_status(
        self,
        document_id,
        processing_status,
        chunk_count=None
    ):
        """
        Update the processing status of a document.

        If chunk_count is provided, update the chunk count
        at the same time.
        """

        connection = self.get_connection()

        cursor = connection.cursor()

        if chunk_count is None:

            cursor.execute(
                """
                UPDATE documents
                SET processing_status = ?
                WHERE id = ?
                """,
                (
                    processing_status,
                    document_id
                )
            )

        else:

            cursor.execute(
                """
                UPDATE documents
                SET
                    processing_status = ?,
                    chunk_count = ?
                WHERE id = ?
                """,
                (
                    processing_status,
                    chunk_count,
                    document_id
                )
            )

        connection.commit()

        connection.close()

