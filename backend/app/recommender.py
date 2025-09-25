from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from .embedding import EmbeddingModel

class FingerprintRecommender:
    def __init__(self, df: pd.DataFrame, embedding_model: str = 'bert'):
        req = {"ResearcherName","FieldOfResearch","TopicName","Percentage"}
        if not req.issubset(set(df.columns)):
            missing = req - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()
        df["Percentage"] = df["Percentage"].astype(float) / 100.0

        pivot = df.pivot_table(index="ResearcherName",
                               columns="TopicName",
                               values="Percentage",
                               aggfunc="mean",
                               fill_value=0.0)

        self.researchers = pivot.index.to_list()
        self.topics = pivot.columns.to_list()
        self.matrix = pivot.values.astype(float)

        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.matrix_norm = self.matrix / norms

        mode_map: Dict[str, str] = {}
        for name, sub in df.groupby("ResearcherName"):
            m = sub["FieldOfResearch"].mode(dropna=True)
            mode_map[name] = m.iloc[0] if not m.empty else None
        self.field_map = mode_map

        self.topic_array = np.array(self.topics)

        self.embedder = EmbeddingModel(model_name=embedding_model)
        self.researcher_texts = [f"{name} {self.field_map.get(name, '')} " + ' '.join([t for t in self.topics if pivot.loc[name, t] > 0]) for name in self.researchers]
        self.researcher_embeddings = self.embedder.encode(self.researcher_texts)
        self.topic_embeddings = self.embedder.encode(self.topics)

    def suggest_topics(self, q: str, k: int = 5) -> List[str]:
        ql = q.lower()
        hits = [t for t in self.topics if ql in t.lower()]
        if hits:
            return hits[:k]
        import difflib
        return difflib.get_close_matches(q, self.topics, n=k, cutoff=0.0)

    def topic_vector(self, topic: str) -> Tuple[np.ndarray, str]:
        for t in self.topics:
            if t.lower() == topic.lower():
                vec = np.zeros((len(self.topics),), dtype=float)
                idx = self.topics.index(t)
                vec[idx] = 1.0
                return vec, t
        suggestions = self.suggest_topics(topic, k=1)
        if suggestions:
            t = suggestions[0]
            vec = np.zeros((len(self.topics),), dtype=float)
            idx = self.topics.index(t)
            vec[idx] = 1.0
            return vec, t
        raise ValueError("No matching topic found")

    def recommend(self, topic: str, topk: int = 10, metric: str = 'cosine'):
        topic_emb = self.embedder.encode(topic)[0]

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

        def hamming_distance(a, b):
            return 1.0 - np.mean(a != b)

        def kl_divergence(a, b):
            a = np.clip(a, 1e-9, 1)
            b = np.clip(b, 1e-9, 1)
            return -np.sum(a * np.log(a / b))

        def minkowski_distance(a, b, p=2):
            return -np.linalg.norm(a - b, ord=p)

        def jaccard_similarity(a, b):
            a_bin = a > 0
            b_bin = b > 0
            intersection = np.logical_and(a_bin, b_bin).sum()
            union = np.logical_or(a_bin, b_bin).sum()
            return intersection / (union + 1e-12)

        metric_funcs = {
            'cosine': cosine_similarity,
            'hamming': hamming_distance,
            'kl': kl_divergence,
            'minkowski': minkowski_distance,
            'jaccard': jaccard_similarity,
        }
        metric = metric.lower()
        if metric not in metric_funcs:
            metric_func = cosine_similarity
        else:
            metric_func = metric_funcs[metric]

        scores = np.array([metric_func(emb, topic_emb) for emb in self.researcher_embeddings])
        order = np.argsort(-scores)[:topk]
        topic_scores = np.array([metric_func(emb, topic_emb) for emb in self.topic_embeddings])
        matched_idx = np.argmax(topic_scores)
        matched = self.topics[matched_idx]
        return order.tolist(), scores[order].tolist(), matched

    def top_topics_for_researcher(self, r_idx: int, n: int = 5):
        row = self.matrix[r_idx, :]
        if row.sum() == 0:
            return []
        top_idx = np.argsort(-row)[:n]
        return [f"{self.topic_array[i]} ({row[i]:.2f})" for i in top_idx]
