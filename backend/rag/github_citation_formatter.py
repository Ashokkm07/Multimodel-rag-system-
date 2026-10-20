class GitHubCitationFormatter:

    @staticmethod
    def format_citation(source):

        source_type = source.get(
            "source_type",
            ""
        )

        file_name = source.get(
            "file",
            ""
        )

        name = source.get(
            "name",
            ""
        )

        start_line = source.get(
            "start_line",
            ""
        )

        end_line = source.get(
            "end_line",
            ""
        )

        # -----------------------------------------
        # Repository metadata
        # -----------------------------------------

        if source_type == "repository_metadata":

            return {
                "display": (
                    "📦 Repository metadata"
                ),
                "source_type": source_type,
                "file": "",
                "symbol": "",
                "start_line": "",
                "end_line": ""
            }

        # -----------------------------------------
        # File citation
        # -----------------------------------------

        if (
            source_type == "metadata_file"
            and file_name
        ):

            return {
                "display": (
                    f"📄 {file_name}"
                ),
                "source_type": source_type,
                "file": file_name,
                "symbol": "",
                "start_line": "",
                "end_line": ""
            }

        # -----------------------------------------
        # Function / method citation
        # -----------------------------------------

        if source_type in {
            "metadata_function",
            "semantic"
        }:

            if (
                name
                and start_line
                and end_line
            ):

                return {
                    "display": (
                        f"📄 {file_name}\n"
                        f"🔧 {name}()\n"
                        f"📍 Lines "
                        f"{start_line}–"
                        f"{end_line}"
                    ),
                    "source_type": source_type,
                    "file": file_name,
                    "symbol": name,
                    "start_line": start_line,
                    "end_line": end_line
                }

        # -----------------------------------------
        # Class citation
        # -----------------------------------------

        if source_type == "metadata_class":

            return {
                "display": (
                    f"📄 {file_name}\n"
                    f"🏷️ {name}\n"
                    f"📍 Lines "
                    f"{start_line}–"
                    f"{end_line}"
                ),
                "source_type": source_type,
                "file": file_name,
                "symbol": name,
                "start_line": start_line,
                "end_line": end_line
            }

        # -----------------------------------------
        # Generic source citation
        # -----------------------------------------

        citation = {
            "display": "",
            "source_type": source_type,
            "file": file_name,
            "symbol": name,
            "start_line": start_line,
            "end_line": end_line
        }

        parts = []

        if file_name:
            parts.append(
                f"📄 {file_name}"
            )

        if name:
            parts.append(
                f"🔧 {name}"
            )

        if start_line and end_line:
            parts.append(
                f"📍 Lines "
                f"{start_line}–"
                f"{end_line}"
            )

        citation["display"] = "\n".join(
            parts
        )

        return citation

    @staticmethod
    def format_sources(sources):

        formatted = []

        seen = set()

        for source in sources:

            citation = (
                GitHubCitationFormatter
                .format_citation(
                    source
                )
            )

            key = (
                citation["file"],
                citation["symbol"],
                citation["start_line"],
                citation["end_line"],
                citation["source_type"]
            )

            if key in seen:
                continue

            seen.add(key)

            formatted.append(
                citation
            )

        return formatted