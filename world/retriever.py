"""
Simple retriever: ranks a world's documents against the question using
word-overlap (Jaccard-like) scoring. This intentionally is NOT a real
embedding retriever -- the point of phase 1 is to validate the RL loop,
not to build SOTA retrieval. Swap this for a real embedding/BM25
retriever later without touching the env.
"""
import re
from dataclasses import dataclass
from typing import List, Tuple

from world.generator import Document, World


@dataclass
class RankedDoc:
    doc: Document
    score: float


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def rank_documents(world: World) -> List[RankedDoc]:
    """Rank ALL of a world's documents by overlap with the question.
    Adds small random-ish noise via partial credit so it's not a perfect
    oracle (realistic retrieval is imperfect)."""
    q_tokens = _tokenize(world.question)
    ranked = []
    for doc in world.documents:
        d_tokens = _tokenize(doc.text)
        overlap = len(q_tokens & d_tokens)
        score = overlap / max(1, len(q_tokens))
        ranked.append(RankedDoc(doc=doc, score=score))
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


def retrieve_top_k(world: World, k: int) -> List[RankedDoc]:
    ranked = rank_documents(world)
    return ranked[:k]


def build_prompt(question: str, ranked_docs: List[RankedDoc]) -> str:
    """Build the LLM prompt from a list of (already-selected) ranked docs.
    Docs are tagged [DOCi] so providers (and the mock) can count context size."""
    doc_lines = []
    for i, rd in enumerate(ranked_docs):
        doc_lines.append(f"[DOC{i}] {rd.doc.text}")
    context = "\n".join(doc_lines) if doc_lines else "(no documents retrieved)"
    return (
        "You are answering a question using ONLY the documents below. "
        "If the answer is present, respond with just the value after ANSWER_MARKER:. "
        "If not present, respond UNKNOWN.\n\n"
        f"Documents:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )