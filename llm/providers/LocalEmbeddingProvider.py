from sentence_transformers import SentenceTransformer, models

from helper.ingestion_errors import ModelInitializationError
from llm.embedding_resource import EmbeddingResource


LOCAL_MAX_SEQ_LENGTH = 128
LOCAL_POOLING_MODE = "mean"
LOCAL_NORMALIZE_EMBEDDINGS = True


class LocalEmbeddingProvider(EmbeddingResource):
    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_size: int | None = None,
    ):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty model identifier.")
        model_id = model_id.strip()
        try:
            model = self._init_model(model_id)
        except Exception as exc:
            raise ModelInitializationError("The local embedding model could not be initialized.") from exc
        super().__init__(
            model,
            provider="LOCAL_EMBEDDING",
            model_id=model_id,
            pipeline="transformer-mean-normalize",
            normalize_embeddings=LOCAL_NORMALIZE_EMBEDDINGS,
            expected_dimension=embedding_size,
        )

    @staticmethod
    def _init_model(model_id: str):
        transformer = models.Transformer(model_id, max_seq_length=LOCAL_MAX_SEQ_LENGTH)
        pooling = models.Pooling(
            transformer.get_word_embedding_dimension(), pooling_mode=LOCAL_POOLING_MODE
        )
        modules = [transformer, pooling]
        if LOCAL_NORMALIZE_EMBEDDINGS:
            modules.append(models.Normalize())
        return SentenceTransformer(modules=modules)
