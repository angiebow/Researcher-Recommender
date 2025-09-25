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

    def recommend(self, topic: str, topk: int = 10):
        topic_emb = self.embedder.encode(topic)[0]
        scores = self.researcher_embeddings @ topic_emb / (np.linalg.norm(self.researcher_embeddings, axis=1) * np.linalg.norm(topic_emb) + 1e-12)
        order = np.argsort(-scores)[:topk]
        topic_sim = self.topic_embeddings @ topic_emb / (np.linalg.norm(self.topic_embeddings, axis=1) * np.linalg.norm(topic_emb) + 1e-12)
        matched_idx = np.argmax(topic_sim)
        matched = self.topics[matched_idx]
        return order.tolist(), scores[order].tolist(), matched

    def top_topics_for_researcher(self, r_idx: int, n: int = 5):
        row = self.matrix[r_idx, :]
        if row.sum() == 0:
            return []
        top_idx = np.argsort(-row)[:n]
        return [f"{self.topic_array[i]} ({row[i]:.2f})" for i in top_idx]
