import re


class GitHubQuestionAnalyzer:

    FILE_COUNT_PATTERNS = [
        r"\bhow many files\b",
        r"\bnumber of files\b",
        r"\btotal files\b",
        r"\bfiles are there\b",
        r"\bcount of files\b"
    ]

    FILE_LIST_PATTERNS = [
        r"\bwhat files\b",
        r"\bwhich files\b",
        r"\blist files\b",
        r"\blist all files\b",
        r"\bfile names\b",
        r"\bfiles present\b",
        r"\bfiles in the repository\b"
    ]

    LANGUAGE_PATTERNS = [
        r"\bwhat programming languages\b",
        r"\bwhich programming languages\b",
        r"\bwhat languages\b",
        r"\bwhich languages\b",
        r"\blanguages used\b",
        r"\bprogramming language\b"
    ]

    MAIN_FILE_PATTERNS = [
        r"\bwhat is the main file\b",
        r"\bwhich is the main file\b",
        r"\bmain file\b",
        r"\bentry point\b",
        r"\bapplication entry point\b"
    ]

    FUNCTION_LIST_PATTERNS = [
        r"\bwhat functions\b",
        r"\bwhich functions\b",
        r"\blist functions\b",
        r"\bfunctions implemented\b",
        r"\bfunctions available\b",
        r"\bwhat methods\b",
        r"\bwhich methods\b"
    ]

    CLASS_LIST_PATTERNS = [
        r"\bwhat classes\b",
        r"\bwhich classes\b",
        r"\blist classes\b",
        r"\bclasses implemented\b",
        r"\bclasses available\b"
    ]

    FILE_CONTENT_PATTERNS = [
        r"\bcontent of\b",
        r"\bcontents of\b",
        r"\bshow.*file\b",
        r"\bread.*file\b",
        r"\bwhat is inside\b.*\bfile\b"
    ]

    ARCHITECTURE_PATTERNS = [
        r"\barchitecture\b",
        r"\bproject architecture\b",
        r"\bapplication architecture\b",
        r"\bcomplete project\b",
        r"\boverall project\b",
        r"\bdata flow\b",
        r"\bworkflow\b",
        r"\bhow does the project work\b",
        r"\bhow does the application work\b"
    ]

    CODE_PATTERNS = [
        r"\bwhere\b.*\bimplemented\b",
        r"\bwhere\b.*\bdefined\b",
        r"\bwhere\b.*\bhandled\b",
        r"\bhow does\b",
        r"\bhow is\b",
        r"\bexplain\b",
        r"\bimplementation\b",
        r"\bcode\b",
        r"\bfunction\b",
        r"\bclass\b",
        r"\bmethod\b"
    ]

    def _matches(
        self,
        question,
        patterns
    ):
        question = question.lower()

        return any(
            re.search(
                pattern,
                question
            )
            for pattern in patterns
        )

    def analyze(self, question):

        if not question or not question.strip():

            return {
                "type": "invalid",
                "question": question
            }

        question = question.strip()

        if self._matches(
            question,
            self.FILE_COUNT_PATTERNS
        ):
            question_type = "file_count"

        elif self._matches(
            question,
            self.FILE_LIST_PATTERNS
        ):
            question_type = "file_list"

        elif self._matches(
            question,
            self.LANGUAGE_PATTERNS
        ):
            question_type = "language_list"

        elif self._matches(
            question,
            self.MAIN_FILE_PATTERNS
        ):
            question_type = "main_file"

        elif self._matches(
            question,
            self.FUNCTION_LIST_PATTERNS
        ):
            question_type = "function_list"

        elif self._matches(
            question,
            self.CLASS_LIST_PATTERNS
        ):
            question_type = "class_list"

        elif self._matches(
            question,
            self.FILE_CONTENT_PATTERNS
        ):
            question_type = "file_content"

        elif self._matches(
            question,
            self.ARCHITECTURE_PATTERNS
        ):
            question_type = "architecture"

        elif self._matches(
            question,
            self.CODE_PATTERNS
        ):
            question_type = "code"

        else:
            question_type = "code"

        return {
            "type": question_type,
            "question": question
        }