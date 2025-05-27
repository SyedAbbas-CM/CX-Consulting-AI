from typing import List


class ChromaEmbeddingWrapper:
    def __init__(self, emb_mgr):
        # emb_mgr.get_embedding_function returns a Callable[[List[str]], List[List[float]]]
        self._emb = emb_mgr.get_embedding_function()

    # Chroma expects __call__(self, input: Documents) -> Embeddings
    # where Documents = List[str], Embeddings = List[List[float]]
    def __call__(self, input: List[str]) -> List[List[float]]:
        # Ensure the wrapped function gets the list of strings
        return self._emb(input)
