"""
Repository and Storage Layer for Question Corpus.
Supports exact string index, dynamic tag queries, vector embedding cache, and serialization.
"""

from __future__ import annotations
import os
import json
from typing import Dict, List, Optional, Any
from .models import Question, normalize_question_text

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "questions_corpus.json")

class QuestionRepository:
    def __init__(self, storage_path: str = DEFAULT_DATA_PATH):
        self.storage_path = storage_path
        self._questions: Dict[str, Question] = {}
        self._normalized_index: Dict[str, str] = {}  # normalized_text -> question_id
        self.load()

    def add_question(self, question: Question) -> Question:
        self._questions[question.id] = question
        self._normalized_index[question.normalized_text] = question.id
        return question

    def get_question(self, question_id: str) -> Optional[Question]:
        return self._questions.get(question_id)

    def get_by_normalized_text(self, normalized_text: str) -> Optional[Question]:
        qid = self._normalized_index.get(normalized_text)
        if qid:
            return self._questions.get(qid)
        return None

    def list_questions(self, concept: Optional[str] = None, tag_filter: Optional[Dict[str, Any]] = None) -> List[Question]:
        res = list(self._questions.values())
        if concept:
            res = [q for q in res if q.concept.lower() == concept.lower()]
        if tag_filter:
            res = [
                q for q in res
                if all(q.tags.get(k) == v for k, v in tag_filter.items())
            ]
        return res

    def get_all(self) -> List[Question]:
        """Returns all questions in repository, auto-seeding Q1-Q10 if empty."""
        if not self._questions:
            for sq in self.get_seed_test_corpus():
                self.add_question(sq)
        return list(self._questions.values())

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = [q.model_dump() for q in self._questions.values()]
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    for item in raw:
                        q = Question.model_validate(item)
                        self.add_question(q)
            except Exception as e:
                print(f"[Warning] Failed to load questions from {self.storage_path}: {e}")

    def clear(self) -> None:
        self._questions.clear()
        self._normalized_index.clear()
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)

    @classmethod
    def get_seed_test_corpus(cls) -> List[Question]:
        """
        Returns the official assignment test corpus Q1-Q10 with dynamic tags.
        Demonstrates arbitrary tag key-value pairs while verifying content-based deduplication.
        """
        seed_data = [
            ("Q1", "What is reflection of light?", {"difficulty": "easy", "bloom": "remember", "source": "textbook_ch1"}),
            ("Q2", "Define the reflection of light.", {"difficulty": "easy", "bloom": "understand", "author": "Dr. Smith"}),
            ("Q3", "What do you understand by the reflection of light?", {"difficulty": "medium", "exam": "midterm"}),
            ("Q4", "Explain the meaning of light reflection.", {"difficulty": "easy", "bloom": "explain", "curriculum": "CBSE"}),
            ("Q5", "State the laws of reflection of light.", {"difficulty": "medium", "bloom": "state", "importance": "high"}),
            ("Q6", "What is the angle of reflection?", {"difficulty": "easy", "concept_unit": "angles"}),
            ("Q7", "The angle of incidence is 30°. What is the angle of reflection?", {"difficulty": "hard", "problem_type": "numerical", "marks": 2}),
            ("Q8", "What is reflection?", {"difficulty": "introductory", "scope": "general_physics"}),
            ("Q9", "What is the difference between regular and diffuse reflection?", {"difficulty": "medium", "bloom": "compare"}),
            ("Q10", "Define the reflection of light.", {"difficulty": "easy", "author": "Prof. Rao", "exam": "final_2024"}),
        ]
        questions = []
        for qid, text, tags in seed_data:
            q = Question(
                id=qid,
                text=text,
                concept="reflection of light",
                tags=tags,
                source="seed"
            )
            questions.append(q)
        return questions
