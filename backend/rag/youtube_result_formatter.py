
class YouTubeResultFormatter:

    @staticmethod
    def format_timestamp(seconds):
        """
        Convert seconds into HH:MM:SS or MM:SS format.
        """

        seconds = int(float(seconds))

        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60

        if hours > 0:

            return (
                f"{hours:02d}:"
                f"{minutes:02d}:"
                f"{remaining_seconds:02d}"
            )

        return (
            f"{minutes:02d}:"
            f"{remaining_seconds:02d}"
        )

    @staticmethod
    def format_sources(sources):
        """
        Format YouTube sources for display.

        Each source contains the video URL and
        timestamp information.
        """

        formatted_sources = []

        for source in sources:

            start_time = source.get(
                "start_time",
                0
            )

            end_time = source.get(
                "end_time",
                start_time
            )

            formatted_sources.append({
                "type": "youtube",
                "source": source.get(
                    "source",
                    ""
                ),
                "video_id": source.get(
                    "video_id",
                    ""
                ),
                "document_id": source.get(
                    "document_id",
                    ""
                ),
                "start_time": start_time,
                "end_time": end_time,
                "timestamp": (
                    f"{YouTubeResultFormatter.format_timestamp(start_time)}"
                    f" - "
                    f"{YouTubeResultFormatter.format_timestamp(end_time)}"
                ),
                "distance": source.get(
                    "distance",
                    0
                )
            })

        return formatted_sources

    @staticmethod
    def format_result(answer, sources):
        """
        Create a consistent YouTube RAG response.
        """

        return {
            "answer": answer,
            "sources": YouTubeResultFormatter.format_sources(
                sources
            )
        }

