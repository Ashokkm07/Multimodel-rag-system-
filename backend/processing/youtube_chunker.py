from typing import List, Dict, Any


class YouTubeChunker:

    def __init__(
        self,
        max_chars=1200,
        overlap_segments=2,
        min_chars=300
    ):
        self.max_chars = max_chars
        self.overlap_segments = overlap_segments
        self.min_chars = min_chars

    def _create_chunk(
        self,
        segments,
        chunk_number
    ):
        """
        Create one chunk from consecutive transcript
        segments while preserving the exact time range.
        """

        if not segments:
            return None

        texts = []

        for segment in segments:

            text = segment.get(
                "text",
                ""
            ).strip()

            if text:
                texts.append(text)

        if not texts:
            return None

        first_metadata = segments[0].get(
            "metadata",
            {}
        )

        last_metadata = segments[-1].get(
            "metadata",
            {}
        )

        start_time = float(
            first_metadata.get(
                "start_time",
                0
            )
        )

        end_time = float(
            last_metadata.get(
                "end_time",
                start_time
            )
        )

        metadata = first_metadata.copy()

        metadata["start_time"] = start_time
        metadata["end_time"] = end_time

        metadata["segment_start"] = (
            first_metadata.get(
                "segment",
                ""
            )
        )

        metadata["segment_end"] = (
            last_metadata.get(
                "segment",
                ""
            )
        )

        metadata["chunk"] = chunk_number

        metadata["chunk_segment_count"] = len(
            segments
        )

        return {
            "text": " ".join(texts),
            "metadata": metadata
        }

    def chunk(
        self,
        documents
    ):
        """
        Group consecutive transcript segments into
        timestamp-aware semantic chunks.

        Each chunk preserves the exact start/end
        timestamps of the segments it contains.
        """

        chunks = []

        current_segments = []
        current_length = 0
        chunk_number = 1

        for document in documents:

            text = document.get(
                "text",
                ""
            ).strip()

            if not text:
                continue

            segment_length = len(text)

            # -------------------------------------------------
            # Add segment to current chunk
            # -------------------------------------------------

            if (
                current_segments
                and
                current_length + segment_length + 1
                > self.max_chars
            ):

                chunk = self._create_chunk(
                    current_segments,
                    chunk_number
                )

                if chunk:
                    chunks.append(chunk)
                    chunk_number += 1

                # ---------------------------------------------
                # Preserve overlap from previous chunk
                # ---------------------------------------------

                if self.overlap_segments > 0:

                    current_segments = (
                        current_segments[
                            -self.overlap_segments:
                        ]
                    )

                    current_length = sum(
                        len(
                            item.get(
                                "text",
                                ""
                            )
                        )
                        for item in current_segments
                    )

                else:

                    current_segments = []
                    current_length = 0

            current_segments.append(
                document
            )

            current_length += (
                segment_length + 1
            )

        # -----------------------------------------------------
        # Final chunk
        # -----------------------------------------------------

        if current_segments:

            chunk = self._create_chunk(
                current_segments,
                chunk_number
            )

            if chunk:
                chunks.append(chunk)

        # -----------------------------------------------------
        # Remove extremely tiny chunks where possible
        # -----------------------------------------------------

        if len(chunks) <= 1:
            return chunks

        filtered_chunks = []

        for index, chunk in enumerate(
            chunks
        ):

            text_length = len(
                chunk["text"]
            )

            # Keep first/last chunk even if small
            is_first = index == 0
            is_last = index == len(chunks) - 1

            if (
                text_length < self.min_chars
                and not is_first
                and not is_last
            ):
                continue

            filtered_chunks.append(
                chunk
            )

        # Re-number after filtering

        for index, chunk in enumerate(
            filtered_chunks,
            start=1
        ):

            chunk["metadata"]["chunk"] = index

        return filtered_chunks


def chunk_youtube_transcript(
    documents,
    chunk_size=1200,
    chunk_overlap=2
):
    """
    Backward-compatible wrapper for the existing
    YouTube ingestion pipeline.
    """

    chunker = YouTubeChunker(
        max_chars=chunk_size,
        overlap_segments=chunk_overlap
    )

    return chunker.chunk(
        documents
    )