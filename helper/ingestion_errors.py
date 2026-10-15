class ProcessingError(Exception):
    pass


class InvalidDocumentError(ProcessingError):
    pass


class EmptyContentError(InvalidDocumentError):
    pass


class ModelInitializationError(ProcessingError):
    pass


class TranscriptionError(ProcessingError):
    pass


class IndexingError(ProcessingError):
    pass


class InvalidAssetError(ProcessingError):
    pass


class AssetNotFoundError(ProcessingError):
    pass


class ChunkingError(ProcessingError):
    pass


class EmbeddingError(ProcessingError):
    pass


class StorageError(ProcessingError):
    pass
