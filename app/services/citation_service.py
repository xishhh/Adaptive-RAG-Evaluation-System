from app.models.responses import ChunkResult


def build_citations(
    chunks: list[ChunkResult],
    min_relevance_score: float = 0.0,
) -> list[ChunkResult]:
    if not chunks:
        return []

    filtered = [c for c in chunks if c.relevance_score >= min_relevance_score]
    seen: set[str] = set()
    deduplicated: list[ChunkResult] = []
    for chunk in filtered:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            deduplicated.append(chunk)
    deduplicated.sort(key=lambda c: c.relevance_score, reverse=True)
    return deduplicated


def format_context_block(chunks: list[ChunkResult]) -> str:
    if not chunks:
        return "No relevant context found."

    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        page_info = f" | page {chunk.page_number}" if chunk.page_number else ""
        header = f"[Source {i} | {chunk.document_name}{page_info}]"
        blocks.append(f"{header}\n{chunk.chunk_text}")

    return "\n\n".join(blocks)
