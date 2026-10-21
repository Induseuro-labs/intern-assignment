"""Test data: the section-6 corpus plus a held-out labelled set for evaluation."""

TAGS = {"subject": "physics", "chapter": "light", "topic": "reflection"}

# id, text, expected label when checked against the questions *before* it
CORPUS = [
    ("Q1", "What is reflection of light?", "unique"),
    ("Q2", "Define the reflection of light.", "semantic_duplicate"),
    ("Q3", "What do you understand by the reflection of light?", "semantic_duplicate"),
    ("Q4", "Explain the meaning of light reflection.", "semantic_duplicate"),
    ("Q5", "State the laws of reflection of light.", "unique"),
    ("Q6", "What is the angle of reflection?", "unique"),
    ("Q7", "The angle of incidence is 30°. What is the angle of reflection?", "unique"),
    ("Q8", "What is reflection?", "needs_review"),
    ("Q9", "What is the difference between regular and diffuse reflection?", "unique"),
    ("Q10", "Define the reflection of light.", "exact_duplicate"),
]

# Held-out candidates checked against the full Q1-Q10 corpus. Written separately
# from the rules so the evaluation is not just the training examples again.
HELD_OUT = [
    # exact (formatting / punctuation / case)
    ("WHAT IS REFLECTION OF LIGHT??", "exact_duplicate"),
    ("what is   reflection of light", "exact_duplicate"),
    ("State the laws of reflection of light", "exact_duplicate"),
    # semantic paraphrases
    ("What is meant by reflection of light?", "semantic_duplicate"),
    ("Define reflection of light.", "semantic_duplicate"),
    ("Briefly explain what reflection of light is.", "semantic_duplicate"),
    ("Distinguish between regular and diffuse reflection.", "semantic_duplicate"),
    ("Differentiate between diffuse and regular reflection.", "semantic_duplicate"),
    ("Write down the laws of reflection of light.", "semantic_duplicate"),
    ("What does the term angle of reflection mean?", "semantic_duplicate"),
    ("If the angle of incidence is 45 degrees, what will the angle of reflection be?", "semantic_duplicate"),
    ("Explain reflection.", "semantic_duplicate"),
    # genuinely different questions on the same subject (incl. hard negatives)
    ("What is refraction of light?", "unique"),
    ("What is the angle of incidence?", "unique"),
    ("What is total internal reflection?", "unique"),
    ("What is a plane mirror?", "unique"),
    ("Why is the image in a plane mirror laterally inverted?", "unique"),
    ("Give two examples of diffuse reflection in daily life.", "unique"),
    ("How is an image formed by a concave mirror?", "unique"),
    ("What are the uses of a periscope?", "unique"),
    ("What is the normal at the point of incidence?", "unique"),
    ("How does a smooth surface differ from a rough surface in reflecting light?", "unique"),
]
