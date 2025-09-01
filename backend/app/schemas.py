from pydantic import BaseModel
from typing import List, Optional

class RecommendationItem(BaseModel):
    researcher: str
    field: Optional[str] = None
    score: float
    top_topics: List[str] = []

class RecommendationResponse(BaseModel):
    query_topic: str
    matched_topic: str
    total_candidates: int
    results: List[RecommendationItem]

class TopicListResponse(BaseModel):
    topics: List[str]
