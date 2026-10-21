# Question 2 demo log

Embedder: `sentence-transformers/all-MiniLM-L6-v2`; LLM judge: off

## Independent duplicate detection

| candidate | verdict | matched | why |
|---|---|---|---|
| define the REFLECTION of light | **exact_duplicate** | Define the reflection of light. | identical after normalisation |
| What is meant by reflection of light? | **semantic_duplicate** | What is reflection of light? | same intent 'definition' and concepts ['light', 'reflect'] |
| Briefly explain what reflection of light is. | **semantic_duplicate** | What is reflection of light? | same intent 'definition' and concepts ['light', 'reflect'] |
| Differentiate between diffuse and regular reflection. | **semantic_duplicate** | What is the difference between regular and diffuse reflection? | same intent 'difference' and concepts ['diffuse', 'reflect', 'regular'] |
| If the angle of incidence is 45°, find the angle of reflection. | **semantic_duplicate** | The angle of incidence is 30°. What is the angle of reflection? | same intent 'numerical' and concepts ['angle', 'incidence', 'reflect'] |
| Explain reflection in general terms. | **needs_review** | What is reflection? | embedding cosine 0.89 in review band |
| What is refraction of light? | **exact_duplicate** | What is refraction of light? | identical after normalisation |
| What is the angle of incidence? | **unique** | The angle of incidence is 30°. What is the angle of reflection? | different intent (definition vs numerical) |
| What is total internal reflection? | **unique** | What is reflection? | embedding cosine 0.73 |
| Why does a plane mirror form a laterally inverted image? | **unique** | What is the angle of reflection? | different intent (reason vs definition) |

## Generation with avoidance + safeguard

LLM backend: `scripted-offline`, rounds: 2

### Prompt sent in round 1 (bounded context, not the whole bank)

```text
You write exam questions. Topic: reflection of light

The question bank already holds 10 questions in this scope. Do NOT produce any question
that asks the same thing as an existing one, even in different words (e.g. "Define X" duplicates
"What is X?"). Each new question must test a concept or angle not yet covered.

Existing questions closest to this topic:
- Define the reflection of light.
- What is reflection of light?
- Explain the meaning of light reflection.
- What do you understand by the reflection of light?
- State the laws of reflection of light.
- What is reflection?
- What is the angle of reflection?
- What is the difference between regular and diffuse reflection?
- The angle of incidence is 30°. What is the angle of reflection?

Representative questions from elsewhere in the bank (shows breadth):
- (none)

Angles already covered (intent: concept terms):
- definition: light reflect
- laws: law light reflect
- definition: angle reflect
- numerical: angle incidence reflect
- definition: reflect
- difference: diffuse reflect regular

Candidates already rejected as duplicates - do not repeat or rephrase them:
- (none)

Write 6 new, mutually distinct questions. For each give the question, its intent
(definition / laws / numerical / difference / reason / process / examples / application) and the
specific concept it tests.
```

### Accepted (added to bank with origin=generated)

- What is a plane mirror?
- What is the angle of incidence?
- Why is the image formed by a plane mirror laterally inverted?
- What is the normal at the point of incidence?
- Give two examples of regular reflection in daily life.
- How many images are formed when two plane mirrors are placed at 90 degrees to each other?

### Rejected by the safeguard (fed back to the next round)

- Define reflection of light. -> **semantic_duplicate** of "Define the reflection of light." (same intent 'definition' and concepts ['light', 'reflect'])
- Distinguish between regular and diffuse reflection. -> **semantic_duplicate** of "What is the difference between regular and diffuse reflection?" (same intent 'difference' and concepts ['diffuse', 'reflect', 'regular'])
- State the laws of reflection of light! -> **exact_duplicate** of "State the laws of reflection of light." (identical after normalisation)
- If the angle of incidence is 50 degrees, what is the angle of reflection? -> **semantic_duplicate** of "The angle of incidence is 30°. What is the angle of reflection?" (same intent 'numerical' and concepts ['angle', 'incidence', 'reflect'])
- What is meant by a plane mirror? -> **semantic_duplicate** of "What is a plane mirror?" (same intent 'definition' and concepts ['mirror', 'plane'])

### Sent to human review

- (none) 

### Round 2 prompt tail (shows rejected candidates fed back)

```text
 as duplicates - do not repeat or rephrase them:
- Define reflection of light.
- Distinguish between regular and diffuse reflection.
- State the laws of reflection of light!
- If the angle of incidence is 50 degrees, what is the angle of reflection?

Write 1 new, mutually distinct questions. For each give the question, its intent
(definition / laws / numerical / difference / reason / process / examples / application) and the
specific concept it tests.
```
