from sentence_transformers import SentenceTransformer
import torch

class EmbeddingModel:
	MODEL_MAP = {
		'bert': 'sentence-transformers/all-MiniLM-L6-v2',
		'xlnet': 'sentence-transformers/xlnet-base-cased',
		'albert': 'sentence-transformers/paraphrase-albert-small-v2',
		'distilbert': 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
		'mpnet': 'sentence-transformers/all-mpnet-base-v2',
	}

	def __init__(self, model_name: str = 'bert', device: str = None):
		if model_name not in self.MODEL_MAP:
			raise ValueError(f"Model '{model_name}' not supported. Choose from: {list(self.MODEL_MAP.keys())}")
		self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = SentenceTransformer(self.MODEL_MAP[model_name], device=self.device)

	def encode(self, texts):
		"""
		Generate embeddings for a list of texts (or a single text).
		Returns a numpy array of embeddings.
		"""
		if isinstance(texts, str):
			texts = [texts]
		embeddings = self.model.encode(texts, convert_to_numpy=True)
		return embeddings