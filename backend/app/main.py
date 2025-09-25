from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import pandas as pd

from .config import DATA_PATH, TOP_N_TOPIC_PREVIEW
from .schemas import RecommendationResponse, RecommendationItem, TopicListResponse
from .recommender import FingerprintRecommender

app = FastAPI(title="Researcher Recommendation API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = None

@app.on_event("startup")
def startup_event():
    global recommender
    if not os.path.exists(DATA_PATH):
        raise RuntimeError(f"DATA_PATH not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    recommender = FingerprintRecommender(df)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/topics", response_model=TopicListResponse)
def list_topics():
    return TopicListResponse(topics=recommender.topics[:1000])

@app.get("/recommend", response_model=RecommendationResponse)
def recommend(
    topic: str = Query(..., description="Topic name (fuzzy match if not exact)"),
    topk: int = Query(10, ge=1, le=100),
    model: str = Query('mpnet', description="Transformer model to use (e.g. mpnet, bert, xlnet, albert, distilbert)"),
    metric: str = Query('cosine', description="Similarity metric to use (cosine, hamming, kl, minkowski, jaccard)")
):
    global recommender
    try:
        import pandas as pd
        from .config import DATA_PATH
        df = pd.read_csv(DATA_PATH)
        recommender = FingerprintRecommender(df, embedding_model=model)
        order, scores, matched = recommender.recommend(topic, topk=topk, metric=metric)
    except ValueError as e:
        suggestions = recommender.suggest_topics(topic, k=5)
        raise HTTPException(status_code=404, detail={"error": str(e), "suggestions": suggestions})

    results: List[RecommendationItem] = []
    for idx, score in zip(order, scores):
        name = recommender.researchers[idx]
        field = recommender.field_map.get(name)
        top_topics = recommender.top_topics_for_researcher(idx, n=TOP_N_TOPIC_PREVIEW)
        results.append(RecommendationItem(researcher=name, field=field, score=float(score), top_topics=top_topics))

    return RecommendationResponse(
        query_topic=topic,
        matched_topic=matched,
        total_candidates=len(recommender.researchers),
        results=results
    )
