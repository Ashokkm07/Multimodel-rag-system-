from backend.rag.multimodal_generation import MultimodalRAGGenerator


class MultimodalTasks:
    """
    On-demand tasks for the unified multimodal RAG system.

    Supported:
    - Summary
    - Question generation
    """

    def __init__(
        self,
        top_k=5,
        llm=None,
        provider=None,
        model_name=None,
    ):
        self.generator = MultimodalRAGGenerator(
            top_k=top_k,
            llm=llm,
            provider=provider,
            model_name=model_name,
        )

    def _get_evidence(
        self,
        topic,
        source_type="all",
        top_k=None,
    ):
        retrieval = self.generator.router.retrieve(
            topic,
            source_type=source_type,
        )

        if top_k is not None:
            results = retrieval.get("results", [])
            retrieval["results"] = results[:top_k]

        return retrieval.get("results", [])

    def _build_context(
        self,
        results,
        max_chars=6000,
    ):
        if not results:
            return ""

        parts = []
        total_chars = 0

        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source_type = metadata.get("source_type", "unknown")
            source = metadata.get(
                "source",
                metadata.get("filename", "unknown"),
            )

            text = result.get("text", "")

            remaining = max_chars - total_chars

            if remaining <= 0:
                break

            if len(text) > remaining:
                text = text[:remaining]

            part = (
                f"EVIDENCE {index}\n"
                f"Source type: {source_type}\n"
                f"Source: {source}\n\n"
                f"Content:\n{text}\n"
            )

            parts.append(part)
            total_chars += len(text)

        return "\n\n".join(parts)

    def summarize(
        self,
        source_type="all",
    ):
        """
        Generate an on-demand summary from retrieved
        multimodal evidence.
        """

        if source_type == "youtube":
            evidence_top_k = 3
            max_chars = 4000
        else:
            evidence_top_k = 5
            max_chars = 6000

        results = self._get_evidence(
            "main topic important information summary",
            source_type,
            top_k=evidence_top_k,
        )

        if not results:
            return {
                "summary": (
                    "There is not enough indexed information "
                    "to generate a summary."
                ),
                "source_type": source_type,
                "sources": [],
            }

        context = self._build_context(results, max_chars=max_chars)

        prompt = f"""
You are a grounded multimodal RAG summarization system.

Create a concise summary using ONLY the evidence below.

Rules:
1. Do not add outside information.
2. Do not invent facts.
3. Include the main ideas supported by the evidence.
4. Remove repetition.
5. Be concise.
6. Do not fabricate citations.

# EVIDENCE

# {context}

SUMMARY:
"""

        summary = self.generator.llm.generate(prompt)

        return {
            "summary": summary,
            "source_type": source_type,
            "sources": self.generator._build_sources(results),
        }

    def generate_questions(
        self,
        number_of_questions=5,
        source_type="all",
    ):
        """
        Generate questions that can be answered from
        the indexed multimodal evidence.
        """

        if number_of_questions <= 0:
            raise ValueError("number_of_questions must be greater than 0.")

        results = self._get_evidence(
            "important concepts facts details content",
            source_type,
            top_k=5,
        )

        if not results:
            return {
                "questions": [],
                "source_type": source_type,
                "sources": [],
            }

        context = self._build_context(results, max_chars=6000)

        prompt = f"""
You are a grounded multimodal RAG question-generation system.

Generate exactly {number_of_questions} useful questions
that can be answered using ONLY the evidence below.

Rules:
1. Do not ask questions requiring outside knowledge.
2. Every question must be answerable from the evidence.
3. Cover different important parts of the evidence.
4. Do not invent details.
5. Return only the questions.
6. Number them from 1 to {number_of_questions}.

# EVIDENCE

# {context}

QUESTIONS:
"""

        response = self.generator.llm.generate(prompt)
        questions = self._parse_questions(response)

        return {
            "questions": questions,
            "source_type": source_type,
            "sources": self.generator._build_sources(results),
        }

    def _parse_questions(self, text):
        if not text:
            return []

        questions = []

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            if len(line) > 2 and line[0].isdigit():
                if line[1:3] in (". ", ") "):
                    line = line[3:].strip()

            if line.startswith("- "):
                line = line[2:].strip()

            if line:
                questions.append(line)

        return questions