"""Self-hosted models, run in-process on CPU (or GPU if available).

Embedder: BAAI/bge-m3 via sentence-transformers (model card: 1024-dim dense vectors, no query
instruction needed; its sentence-transformers config ends with a Normalize module).
NLI: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (model card: premise/hypothesis pairs;
config id2label = {0: entailment, 1: neutral, 2: contradiction}; 512 tokens; trained on hi and mr).

Both are pinned to a Hugging Face commit and loaded lazily on first use. Install with
`pip install -e '.[models]'`.
"""

from __future__ import annotations

import asyncio
import threading

from app.models.schemas import NLIScore


class _Lazy:
    def __init__(self) -> None:
        self._lock = threading.Lock()  # one load, and one forward pass at a time
        self._model = None

    def _load(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def get(self):
        with self._lock:
            if self._model is None:
                self._model = self._load()
            return self._model


class BGEM3Embedder(_Lazy):
    def __init__(self, model_name: str = "BAAI/bge-m3", revision: str | None = None, max_seq_length: int = 1024,
                 dim: int = 1024, batch_size: int = 8):
        super().__init__()
        self.model_name, self.revision = model_name, revision
        self.max_seq_length, self.dim, self.batch_size = max_seq_length, dim, batch_size
        self.model_version = f"{model_name}@{(revision or 'main')[:12]}"

    def _load(self):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_name, revision=self.revision, device=None)
        model.max_seq_length = self.max_seq_length
        dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        got = dim_fn()
        if got != self.dim:
            raise ValueError(f"{self.model_name} outputs {got}-dim vectors but EMBEDDING_DIM={self.dim}")
        return model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self.get()
        with self._lock:
            vecs = model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, texts)


class MDebertaNLI(_Lazy):
    def __init__(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                 revision: str | None = None, batch_size: int = 8):
        super().__init__()
        self.model_name, self.revision, self.batch_size = model_name, revision, batch_size
        self.model_version = f"{model_name}@{(revision or 'main')[:12]}"

    def _load(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, revision=self.revision)
        model.eval()
        labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
        if not {"entailment", "neutral", "contradiction"} <= set(labels):
            raise ValueError(f"unexpected NLI labels: {model.config.id2label}")
        return tok, model, labels

    def _score(self, pairs: list[tuple[str, str]]) -> list[NLIScore]:
        import torch

        tok, model, labels = self.get()
        out: list[NLIScore] = []
        with self._lock, torch.inference_mode():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                enc = tok(
                    [p for p, _ in batch], [h for _, h in batch],
                    truncation="only_first", max_length=512, padding=True, return_tensors="pt",
                ).to(model.device)
                probs = torch.softmax(model(**enc).logits, dim=-1).tolist()
                out += [
                    NLIScore(
                        entailment=row[labels["entailment"]],
                        neutral=row[labels["neutral"]],
                        contradiction=row[labels["contradiction"]],
                    )
                    for row in probs
                ]
        return out

    async def score(self, pairs: list[tuple[str, str]]) -> list[NLIScore]:
        if not pairs:
            return []
        return await asyncio.to_thread(self._score, pairs)
