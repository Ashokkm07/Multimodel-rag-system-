
import re

from youtube_transcript_api import YouTubeTranscriptApi


class YouTubeLoader:

    def __init__(self):
        """
        Initialize the YouTube Transcript API.
        """

        self.api = YouTubeTranscriptApi()

    # -------------------------------------------------
    # EXTRACT VIDEO ID
    # -------------------------------------------------

    def extract_video_id(self, youtube_url):
        """
        Extract the 11-character YouTube video ID
        from common YouTube URL formats.
        """

        patterns = [
            r"(?:v=)([A-Za-z0-9_-]{11})",
            r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
            r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
            r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})"
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                youtube_url
            )

            if match:

                return match.group(1)

        raise ValueError(
            "Could not extract a valid YouTube video ID "
            "from the provided URL."
        )

    # -------------------------------------------------
    # LOAD TRANSCRIPT
    # -------------------------------------------------

    def load(self, youtube_url):
        """
        Extract a transcript from a YouTube video.

        English is preferred when available.

        If English is not available, the loader automatically
        uses the first available transcript language.

        This allows the system to process videos with:
        - English transcripts
        - Tamil transcripts
        - Other available transcript languages
        """

        video_id = self.extract_video_id(
            youtube_url
        )

        try:

            # -------------------------------------------------
            # FETCH AVAILABLE TRANSCRIPTS
            # -------------------------------------------------

            transcript_list = self.api.list(
                video_id
            )

            print(
                "\nAvailable YouTube transcripts:"
            )

            available_transcripts = []

            for transcript in transcript_list:

                available_transcripts.append(
                    transcript
                )

                print(
                    f"- {transcript.language_code} "
                    f"({transcript.language})"
                )

            if not available_transcripts:

                raise RuntimeError(
                    "No transcripts are available for this video."
                )

            # -------------------------------------------------
            # PREFER ENGLISH
            # -------------------------------------------------

            selected_transcript = None

            for transcript in available_transcripts:

                if transcript.language_code == "en":

                    selected_transcript = transcript

                    print(
                        "\nEnglish transcript selected."
                    )

                    break

            # -------------------------------------------------
            # FALLBACK TO FIRST AVAILABLE LANGUAGE
            # -------------------------------------------------

            if selected_transcript is None:

                selected_transcript = (
                    available_transcripts[0]
                )

                print(
                    "\nEnglish transcript not available."
                )

                print(
                    "Using available transcript: "
                    f"{selected_transcript.language_code} "
                    f"({selected_transcript.language})"
                )

            # -------------------------------------------------
            # FETCH SELECTED TRANSCRIPT
            # -------------------------------------------------

            transcript = selected_transcript.fetch()

        except Exception as error:

            raise RuntimeError(
                f"Could not retrieve YouTube transcript: "
                f"{error}"
            )

        # -----------------------------------------------------
        # CONVERT TRANSCRIPT INTO OUR DOCUMENT FORMAT
        # -----------------------------------------------------

        documents = []

        for index, segment in enumerate(
            transcript,
            start=1
        ):

            text = segment.text.strip()

            if not text:

                continue

            start_time = float(
                segment.start
            )

            duration = float(
                segment.duration
            )

            end_time = (
                start_time +
                duration
            )

            documents.append({
                "text": text,

                "metadata": {
                    "source": youtube_url,

                    "source_type": "youtube",

                    "video_id": video_id,

                    "language": selected_transcript.language_code,

                    "language_name": selected_transcript.language,

                    "start_time": start_time,

                    "end_time": end_time,

                    "segment": index
                }
            })

        return documents

