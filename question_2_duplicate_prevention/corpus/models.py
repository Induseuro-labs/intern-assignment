"""
Domain models for Question Corpus, Metadata Tags, and Duplicate Classification.
"""

from __future__ import annotations
import re
import string
import uuid
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class ClassificationCategory(str, Enum):
    ORIGINAL = "Original"
    EXACT_DUPLICATE = "Exact duplicate"
    SEMANTIC_DUPLICATE = "Semantic duplicate"
    DIFFERENT_QUESTION = "Different question"
    POTENTIALLY_AMBIGUOUS = "Potentially ambiguous; review context"


def normalize_question_text(text: str) -> str:
    """
    Canonically normalizes text for exact and surface comparison:
    - Lowercases text
    - Strips leading/trailing punctuation (question marks, periods)
    - Normalizes internal punctuation and whitespaces
    - Expands common contractions
    """
    text = text.lower().strip()
    # Contraction expansions
    contractions = {
        "what's": "what is",
        "define's": "define is",
        "how's": "how is",
        "doesn't": "does not",
        "don't": "do not",
    }
    for c, expanded in contractions.items():
        text = re.sub(rf"\b{c}\b", expanded, text)
        
    # Remove punctuation except alphanumeric characters and internal dashes/degrees
    text = re.sub(r"[^\w\s°]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


class Question(BaseModel):
    """
    Represents a stored question with arbitrary key-value tags and semantic embeddings.
    """
    id: str = Field(..., description="Unique Question ID (e.g. 'Q1', 'Q2')")
    text: str = Field(..., description="Original raw question string")
    concept: str = Field(default="reflection of light", description="Subject or core concept")
    
    # Arbitrary key-value metadata tags
    tags: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dynamic key-value pairs (e.g. {'subject': 'Physics', 'difficulty': 'easy'})"
    )
    
    source: str = Field("user_corpus", description="'seed', 'user_corpus', or 'llm_generated'")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Derived normalized representation (cached for rapid lookup)
    normalized_text: str = Field("")

    def model_post_init(self, __context: Any) -> None:
        if not self.normalized_text:
            self.normalized_text = normalize_question_text(self.text)


class DuplicateCheckResult(BaseModel):
    """
    Detailed verdict produced by the Independent Duplicate Detector.
    """
    candidate_id: Optional[str] = None
    candidate_text: str
    predicted_classification: ClassificationCategory
    matched_question_id: Optional[str] = None
    matched_question_text: Optional[str] = None
    similarity_score: float = 0.0
    exact_match: bool = False
    confidence: float = 1.0
    stage_resolved: str = "undetermined"  # 'exact_hash', 'lexical_filter', 'semantic_embedding', 'intent_heuristic'
    reasoning: str = ""


class QuestionGenerationRequest(BaseModel):
    subject: str = "Physics"
    concept: str = "reflection of light"
    num_questions: int = 3
    tags: Dict[str, Any] = Field(default_factory=dict)
    avoid_duplicates: bool = True
    context_exemplar_limit: int = 5
