"""
Retrieval module for searching relevant document chunks.

Provides three complementary search strategies:

  search()           — Standard semantic (cosine) search.
  search_mmr()       — Maximal Marginal Relevance: balances relevance with
                       diversity so consecutive results are not near-duplicates.
  search_expanded()  — Multi-query expansion: rewrites the query with synonym
                       variants, runs parallel searches, then merges & re-ranks
                       by a combined semantic + keyword score.
  get_context()      — Formats results into a ready-to-use RAG context string
                       with source attribution.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Dict, Any

from domain.models import SearchResult
from embeddings.base import BaseEmbedding
from vectorstore.base import BaseVectorStore, cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _keyword_score(text: str, query_terms: List[str]) -> float:
    """Fraction of query terms found in *text* (case-insensitive)."""
    if not query_terms:
        return 0.0
    lower = text.lower()
    hits = sum(1 for t in query_terms if t in lower)
    return hits / len(query_terms)


def _expand_query(query: str) -> List[str]:
    """
    Generate lightweight query variants for multi-query expansion.

    Strategy (no LLM required):
      - original query
      - question → statement transformation  ("What is X?" → "X is")
      - key noun-phrases extracted from the query
    """
    variants: List[str] = [query]

    # Strip leading question words to create a declarative variant
    # English and Spanish question words
    declarative = re.sub(
        r"^\s*(what|who|where|when|why|how|which|describe|explain"
        r"|¿qué|qué|¿cuál|cuál|¿cuáles|cuáles|¿dónde|dónde|¿cuándo|cuándo"
        r"|¿cómo|cómo|¿quién|quién|¿quiénes|quiénes|explica|describe|cuéntame)\s+"
        r"(is|are|was|were|do|does|did|can|could|should|would|es|son|fue|era|son|está|están|hay)?\s*",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip(" ?")
    if declarative and declarative.lower() != query.lower():
        variants.append(declarative)

    # Add a "definition / explanation" framing
    core = re.sub(r"[^\w\s]", "", query).strip()
    if len(core.split()) <= 6:
        variants.append(f"definition and explanation of {core}")

    return list(dict.fromkeys(variants))  # deduplicate, preserve order


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class RetrieverException(Exception):
    """Exception raised for retrieval errors."""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DocumentRetriever:
    """
    Orchestrates document retrieval using semantic search.

    Provides the 'R' in RAG (Retrieval-Augmented Generation) with three
    search modes: standard, MMR-diversified, and multi-query-expanded.
    """

    # Default MMR lambda: 0.7 = 70 % relevance, 30 % diversity
    MMR_LAMBDA: float = 0.7

    def __init__(
        self,
        embedder: BaseEmbedding,
        vector_store: BaseVectorStore,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        logger.info(
            "DocumentRetriever initialised — "
            "embedder=%s, vector_store=%s",
            embedder.__class__.__name__,
            vector_store.__class__.__name__,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Standard semantic search.

        Embeds *query*, queries the vector store and returns results sorted
        by cosine similarity (highest first).

        Args:
            query:     Natural-language query string.
            top_k:     Maximum results to return (vector store default if None).
            min_score: Minimum similarity threshold (vector store default if None).
            where:     Metadata filters, e.g. ``{"document_id": "doc1"}``.

        Returns:
            List[SearchResult] ordered by descending score.

        Raises:
            RetrieverException: On any embedding or store failure.
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to search()")
            return []

        logger.info("search | query='%s' top_k=%s min_score=%s", query[:60], top_k, min_score)

        try:
            q_emb = self._embed_query(query)
            results = self.vector_store.search(
                query_embedding=q_emb,
                top_k=top_k,
                min_score=min_score,
                where=where,
            )
            logger.info("search | %d results returned", len(results))
            return results
        except Exception as exc:
            msg = f"search() failed for query '{query[:60]}': {exc}"
            logger.error(msg, exc_info=True)
            raise RetrieverException(msg) from exc

    def search_mmr(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: Optional[int] = None,
        mmr_lambda: Optional[float] = None,
        min_score: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Maximal Marginal Relevance search.

        Retrieves *fetch_k* candidates from the vector store, then
        iteratively selects the result that best balances relevance to the
        query and diversity from already-selected results.

        ``mmr_score = λ · similarity(result, query)
                    - (1-λ) · max_similarity(result, selected)``

        Args:
            query:      Natural-language query string.
            top_k:      Final number of results to return.
            fetch_k:    Candidate pool size (default: ``top_k * 4``).
            mmr_lambda: Trade-off weight ∈ [0, 1].
                        1 = pure relevance, 0 = pure diversity.
                        Defaults to ``MMR_LAMBDA`` (0.7).
            min_score:  Minimum relevance threshold for candidates.
            where:      Metadata filters.

        Returns:
            List[SearchResult] of length ≤ *top_k*, diversity-reranked.

        Raises:
            RetrieverException: On embedding or store failure.
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to search_mmr()")
            return []

        _lambda = mmr_lambda if mmr_lambda is not None else self.MMR_LAMBDA
        _fetch_k = fetch_k if fetch_k is not None else max(top_k * 4, 20)

        logger.info(
            "search_mmr | query='%s' top_k=%d fetch_k=%d lambda=%.2f",
            query[:60], top_k, _fetch_k, _lambda,
        )

        try:
            q_emb = self._embed_query(query)

            # Over-fetch candidates
            candidates = self.vector_store.search(
                query_embedding=q_emb,
                top_k=_fetch_k,
                min_score=min_score,
                where=where,
            )

            if not candidates:
                return []

            # Retrieve stored embeddings for MMR (graceful fallback if unavailable)
            candidate_embeddings: Dict[str, Optional[List[float]]] = {}
            for r in candidates:
                emb = getattr(r.chunk, "embedding", None)
                candidate_embeddings[r.chunk.id] = emb

            selected: List[SearchResult] = []
            selected_embeddings: List[List[float]] = []
            remaining = list(candidates)

            while remaining and len(selected) < top_k:
                best: Optional[SearchResult] = None
                best_score = float("-inf")

                for candidate in remaining:
                    rel_score = candidate.score  # similarity to query

                    # Diversity penalty: max similarity to any already-selected chunk
                    c_emb = candidate_embeddings.get(candidate.chunk.id)
                    if c_emb and selected_embeddings:
                        max_sim = max(
                            cosine_similarity(list(c_emb), list(s_emb)) for s_emb in selected_embeddings
                        )
                    else:
                        max_sim = 0.0

                    mmr_score = _lambda * rel_score - (1.0 - _lambda) * max_sim

                    if mmr_score > best_score:
                        best_score = mmr_score
                        best = candidate

                if best is None:
                    break

                selected.append(best)
                sel_emb = candidate_embeddings.get(best.chunk.id)
                if sel_emb:
                    selected_embeddings.append(sel_emb)
                remaining = [r for r in remaining if r.chunk.id != best.chunk.id]

            logger.info("search_mmr | %d diverse results selected", len(selected))
            return selected

        except Exception as exc:
            msg = f"search_mmr() failed for query '{query[:60]}': {exc}"
            logger.error(msg, exc_info=True)
            raise RetrieverException(msg) from exc

    def search_expanded(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        min_score: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Multi-query expansion with keyword-boosted re-ranking.

        Steps:
          1. Expand *query* into multiple linguistic variants.
          2. Run a semantic search for each variant.
          3. Merge all candidates (deduplicated by chunk id).
          4. Re-rank with a blended score:
             ``score = semantic_weight · best_semantic_score
                     + keyword_weight  · keyword_overlap_score``

        Args:
            query:            Original user query.
            top_k:            Final number of results.
            semantic_weight:  Contribution of vector similarity (0–1).
            keyword_weight:   Contribution of keyword overlap (0–1).
            min_score:        Minimum semantic threshold for individual searches.
            where:            Metadata filters.

        Returns:
            List[SearchResult] of length ≤ *top_k*, re-ranked.

        Raises:
            RetrieverException: On any failure.
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to search_expanded()")
            return []

        logger.info("search_expanded | query='%s' top_k=%d", query[:60], top_k)

        try:
            variants = _expand_query(query)
            query_terms = [
                w for w in re.findall(r"\w+", query.lower()) if len(w) > 2
            ]
            logger.debug("search_expanded | variants=%s", variants)

            # Gather all candidates across variants (deduplicated)
            seen_ids: set = set()
            pool: Dict[str, SearchResult] = {}  # chunk_id → best SearchResult

            for variant in variants:
                v_emb = self._embed_query(variant)
                variant_results = self.vector_store.search(
                    query_embedding=v_emb,
                    top_k=max(top_k * 3, 15),
                    min_score=min_score,
                    where=where,
                )
                for r in variant_results:
                    cid = r.chunk.id
                    if cid not in pool or r.score > pool[cid].score:
                        pool[cid] = r
                    seen_ids.add(cid)

            if not pool:
                return []

            # Re-rank with blended score
            scored: List[tuple[float, SearchResult]] = []
            for r in pool.values():
                kw = _keyword_score(r.chunk.content, query_terms)
                blended = semantic_weight * r.score + keyword_weight * kw
                scored.append((blended, r))

            scored.sort(key=lambda x: x[0], reverse=True)
            final = [r for _, r in scored[:top_k]]

            logger.info(
                "search_expanded | pool=%d candidates → %d results (variants=%d)",
                len(pool), len(final), len(variants),
            )
            return final

        except Exception as exc:
            msg = f"search_expanded() failed for query '{query[:60]}': {exc}"
            logger.error(msg, exc_info=True)
            raise RetrieverException(msg) from exc

    def search_by_document(
        self,
        query: str,
        document_id: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Standard search scoped to a single document.

        Args:
            query:       Natural-language query string.
            document_id: Restrict results to this document.
            top_k:       Maximum results.
            min_score:   Minimum similarity threshold.

        Returns:
            List[SearchResult] from the specified document only.
        """
        logger.info("search_by_document | doc='%s' query='%s'", document_id, query[:50])
        return self.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
            where={"document_id": document_id},
        )

    def get_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: str = "expanded",
        separator: str = "\n\n---\n\n",
        include_sources: bool = True,
    ) -> str:
        """
        Build a formatted context string suitable for passing to an LLM.

        Args:
            query:           Natural-language query string.
            top_k:           Number of chunks to include.
            mode:            Search mode: ``"standard"`` | ``"mmr"`` | ``"expanded"``.
                             Defaults to ``"expanded"`` for best coverage.
            separator:       Separator between chunks.
            include_sources: Prefix each chunk with ``[Source: …]`` metadata.

        Returns:
            Ready-to-use context string, or ``""`` if nothing was found.

        Raises:
            RetrieverException: On retrieval failure.
        """
        _top_k = top_k or 5

        if mode == "mmr":
            results = self.search_mmr(query=query, top_k=_top_k)
        elif mode == "expanded":
            results = self.search_expanded(query=query, top_k=_top_k)
        else:
            results = self.search(query=query, top_k=_top_k)

        if not results:
            logger.warning("get_context | no results for query '%s'", query[:50])
            return ""

        parts: List[str] = []
        for r in results:
            header = ""
            if include_sources:
                doc_name = getattr(r, "document_name", None) or r.chunk.document_id
                header = f"[Source: {doc_name} | score: {r.score:.3f}]\n"
            parts.append(header + r.chunk.content)

        context = separator.join(parts)
        logger.info(
            "get_context | %d chunks, %d chars (mode=%s)", len(results), len(context), mode
        )
        return context

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> List[float]:
        """Embed a query string, raising RetrieverException on failure."""
        try:
            emb = self.embedder.embed_text(query)
            logger.debug("_embed_query | dim=%d", len(emb))
            return emb
        except Exception as exc:
            raise RetrieverException(f"Failed to embed query: {exc}") from exc
