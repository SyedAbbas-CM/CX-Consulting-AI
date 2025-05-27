import asyncio
from typing import Awaitable, Callable, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


class HybridRetriever:
    """
    A plug-and-play retriever that supports multiple hybrid retrieval strategies:

    1. Flat Hybrid Retrieval (BM25 + dense) with Reciprocal Rank Fusion (RRF)
    2. Two-Stage Hybrid + Cross-Encoder re-rank
    3. Query Expansion + Retrieval (stub)
    4. Cluster-Based Routing + Retrieval (stub)
    """

    def __init__(
        self,
        corpus: List[str],
        doc_ids: List[str],
        tokenized_corpus: List[List[str]],
        dense_search_fn: Callable[[str, int], Awaitable[List[Dict]]],
        cross_encoder: Optional[CrossEncoder] = None,
    ):
        """
        Args:
            corpus: List of document/chunk texts in the same order as `doc_ids`.
            doc_ids: Unique identifiers for each document/chunk.
            tokenized_corpus: Pre-tokenized corpus (list of token lists) for BM25.
            dense_search_fn: Function(query: str, k: int) -> List[Dict] with keys 'id','score','text','metadata'.
            cross_encoder: (Optional) CrossEncoder model for re-ranking.
        """
        self.corpus = corpus
        self.doc_ids = doc_ids
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.dense_search_fn = dense_search_fn
        self.cross_encoder = cross_encoder

    async def flat_hybrid_ranking(
        self, query: str, k: int = 10, rrf_k: int = 60
    ) -> List[Dict]:
        """
        Strategy 1: Flat Hybrid Retrieval + RRF

        - Runs BM25 and dense retrieval separately (top-k each).
        - Fuses their ranks via Reciprocal Rank Fusion (RRF).

        Returns a list of top-k hits with fused scores.
        """
        # 1) BM25 retrieval
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:k]
        bm25_hits = [
            {
                "id": self.doc_ids[i],
                "score": float(bm25_scores[i]),
                "text": self.corpus[i],
                "rank": rank,
            }
            for rank, i in enumerate(bm25_indices, start=1)
        ]

        # 2) Dense retrieval
        dense_hits = await self.dense_search_fn(query, k)
        dense_hits_ranked = []
        for idx, hit in enumerate(dense_hits, start=1):
            h = hit.copy()
            h["rank"] = idx
            dense_hits_ranked.append(h)

        # 3) RRF fusion
        score_map = {}
        for hit in bm25_hits + dense_hits_ranked:
            doc_id = hit["id"]
            rank = hit["rank"]
            score_map.setdefault(doc_id, []).append(rank)

        fused = []
        for doc_id, ranks in score_map.items():
            fused.append(
                {
                    "id": doc_id,
                    "rrf_score": sum(1.0 / (rrf_k + r) for r in ranks),
                }
            )

        fused_sorted = sorted(fused, key=lambda x: x["rrf_score"], reverse=True)[:k]

        # 4) Build output
        id_to_text = {h["id"]: h["text"] for h in bm25_hits + dense_hits_ranked}
        return [
            {
                "id": f["id"],
                "score": f["rrf_score"],
                "text": id_to_text[f["id"]],
                "source": "hybrid_rrf",
            }
            for f in fused_sorted
        ]

    async def hybrid_cross_encoder(
        self, query: str, M: int = 50, K: int = 10
    ) -> List[Dict]:
        """
        Strategy 2: Two-Stage Hybrid + Cross-Encoder Re-rank

        1. Use flat hybrid RRF to retrieve top-M candidates.
        2. Re-rank those M candidates with a CrossEncoder model down to top-K.

        Returns a list of top-K hits with cross-encoder scores.
        """
        if self.cross_encoder is None:
            raise ValueError("Cross-encoder model not provided.")

        # Stage 1: top-M via flat hybrid
        candidates = await self.flat_hybrid_ranking(query, k=M)

        # Stage 2: cross-encoder re-rank
        if not candidates:
            return []

        texts = [c["text"] for c in candidates]
        pairs = [[query, t] for t in texts]

        # CrossEncoder.predict is synchronous, run in a thread
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, self.cross_encoder.predict, pairs)

        for c, s in zip(candidates, scores):
            c["cross_score"] = float(s)

        reranked = sorted(candidates, key=lambda x: x["cross_score"], reverse=True)[:K]
        return [
            {
                "id": c["id"],
                "score": c["cross_score"],
                "text": c["text"],
                "source": "hybrid_cross_encoder",
            }
            for c in reranked
        ]

    # Strategy 3: Query Expansion + Retrieval (stub)
    # def query_expansion_retrieval(self, query: str, k: int = 10) -> List[Dict]:
    #     """
    #     Use an LLM to expand the user query, then run flat hybrid on the expanded query.
    #     Useful for broad or sparse queries.
    #     """
    #     # Pseudocode:
    #     # expanded = llm.expand_query(query)
    #     # return self.flat_hybrid_ranking(expanded + " " + query, k)
    #     pass

    # Strategy 4: Cluster-Based Routing + Retrieval (stub)
    # def cluster_routing_retrieval(self, query: str, k: int = 10) -> List[Dict]:
    #     """
    #     Route the query to the nearest embedding clusters, then retrieve within those clusters.
    #     Useful to focus search on relevant topics and reduce candidate set.
    #     """
    #     # Pseudocode:
    #     # query_emb = embed(query)
    #     # nearest_clusters = clusters.find_nearest(query_emb)
    #     # candidates = get_chunks_from_clusters(nearest_clusters)
    #     # return self.flat_hybrid_ranking_on_subset(candidates, query, k)
    #     pass
