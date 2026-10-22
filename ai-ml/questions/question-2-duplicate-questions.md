# Question 2: AI Question Generation and Duplicate Prevention

## 1. Problem Statement

A system stores hundreds or thousands of questions. An LLM is used to generate additional questions for the same subject or category.

Design and implement a solution that minimizes duplicate questions during generation and independently detects duplicates if the prevention mechanism fails.

## 2. Duplicate Avoidance During Generation

Implement a generation workflow that uses the existing question collection to guide the LLM toward producing genuinely new questions.

Address the following:

- How existing questions are selected and provided to the LLM without exceeding context limits.
- How to avoid questions that are identical or substantially similar in meaning, even when their wording differs.
- How the approach scales as the question collection grows.

## 3. Independent Duplicate Detection

Implement a separate mechanism to compare each newly generated question against the existing collection.

The mechanism should identify:

- Exact duplicates, including differences in formatting or punctuation.
- Paraphrased questions that test the same concept.
- Questions that differ in wording or presentation but are effectively equivalent.

This detector must operate independently of generation-time prevention. It serves as a safeguard when prevention fails.

## 4. Tags and Metadata

Questions may have tags represented as key–value pairs. New tag keys and values can be introduced at any time.

Tags may be used to organize or filter questions, but duplicate avoidance and detection must be based on question content rather than tag equality.

## 5. Implementation Requirements

Provide working code for both generation-time avoidance and independent duplicate detection. Explain the algorithms, architecture, technology choices, and trade-offs.

Demonstrate the solution using examples of:

- Exact duplicates
- Semantic duplicates
- Genuinely different questions on the same subject

You may use LLMs, embeddings, vector databases, text similarity algorithms, or other suitable technologies. Explain how you would evaluate the accuracy and performance of the solution.

## 6. Test Data

Use **reflection** and **reflection of light** as test concepts. The dataset below includes exact duplicates, semantic paraphrases, and genuinely different questions on the same subject.

### Question corpus

| ID | Question | Expected classification |
|----|----------|-------------------------|
| Q1 | What is reflection of light? | Original |
| Q2 | Define the reflection of light. | Semantic duplicate of Q1 |
| Q3 | What do you understand by the reflection of light? | Semantic duplicate of Q1 |
| Q4 | Explain the meaning of light reflection. | Semantic duplicate of Q1 |
| Q5 | State the laws of reflection of light. | Different question |
| Q6 | What is the angle of reflection? | Different question |
| Q7 | The angle of incidence is 30°. What is the angle of reflection? | Different question |
| Q8 | What is reflection? | Potentially ambiguous; review context |
| Q9 | What is the difference between regular and diffuse reflection? | Different question |
| Q10 | Define the reflection of light. | Exact duplicate of Q2 |

### How to use this data

- **Generation-time avoidance:** When generating new questions on this topic, the system should steer away from wording and concepts already covered by Q1–Q10 (especially Q1–Q4 and Q10).
- **Independent duplicate detection:** For each new candidate question, classify it against the corpus using the expected classifications above as ground truth when evaluating your solution.
