from backend.rag.multimodal_retriever import (
    MultimodalRetriever,
)


class MultimodalRouter:
    """
    Routes retrieval requests across the unified
    multimodal knowledge collection.

    Supported source types:
        document
        image
        youtube
        all

    Optional source filtering allows the UI to retrieve
    only from the currently selected document/image/video.
    """

    SUPPORTED_SOURCES = {
        "document",
        "image",
        "youtube",
        "all",
    }

    def __init__(self, top_k=5):
        self.retriever = MultimodalRetriever(
            top_k=top_k
        )

    def retrieve(
        self,
        query,
        source_type="all",
        source=None,
        document_id=None,
    ):
        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        source_type = (
            source_type.lower().strip()
        )

        if source_type not in self.SUPPORTED_SOURCES:
            raise ValueError(
                f"Unsupported source type: {source_type}"
            )

        # "all" means do not apply a source_type filter.
        actual_source_type = (
            None
            if source_type == "all"
            else source_type
        )

        results = self.retriever.retrieve(
            query=query,
            source_type=actual_source_type,
            source=source,
            document_id=document_id,
        )

        return {
            "query": query,
            "source_type": source_type,
            "source": source,
            "document_id": document_id,
            "results": results,
            "result_count": len(results),
        }