from qgen import text as T
from qgen.store import QuestionStore


def test_exact_key_ignores_formatting():
    base = T.exact_key("Define the reflection of light.")
    for variant in ["define the reflection of light", "  DEFINE THE REFLECTION OF LIGHT!! ",
                    "Define reflection of light", "Define the   reflection of light?"]:
        assert T.exact_key(variant) == base
    assert T.exact_key("What is reflection?") != T.exact_key("What is reflection of light?")
    assert T.normalize("Angle is 30°.") == T.normalize("angle is 30 degrees")


def test_content_terms_drop_question_wording():
    same = {T.content_terms(q) for q in ["What is reflection of light?", "Define the reflection of light.",
                                         "What do you understand by the reflection of light?",
                                         "Explain the meaning of light reflection."]}
    assert same == {frozenset({"reflect", "light"})}


def test_intent_and_numbers():
    assert T.intent("State the laws of reflection of light.") == "law"
    assert T.intent("What is the difference between regular and diffuse reflection?") == "difference"
    assert T.intent("The angle of incidence is 30°. What is the angle of reflection?") == "calculation"
    assert T.intent("Define the reflection of light.") == "definition"
    assert T.numbers("A 2.5 m tall pole, 1,000 cm away") == ("1000", "2.5")


def test_sub_questions():
    assert T.sub_questions("The angle of incidence is 30°. What is the angle of reflection?") == []
    assert T.sub_questions("State the two laws of reflection of light. Draw a ray diagram for a plane mirror.") == [
        "State the two laws of reflection of light.", "Draw a ray diagram for a plane mirror."]


def test_free_form_tags_and_filtering():
    s = QuestionStore()
    s.add("What is reflection of light?", {"subject": "physics", "chapter": "light"}, qid="Q1")
    s.add("State the laws of reflection of light.", {"subject": "physics", "marks": "2"}, qid="Q5")
    s.add("What is photosynthesis?", {"subject": "biology"}, qid="B1")
    s.set_tags("Q5", {"exam": ["CBSE-2023", "ICSE-2022"], "brand_new_key": "anything"})
    assert s.get("Q5").tags["exam"] == ["CBSE-2023", "ICSE-2022"]
    assert [q.id for q in s.all({"subject": "physics"})] == ["Q1", "Q5"]
    assert [q.id for q in s.all({"exam": "ICSE-2022", "subject": "physics"})] == ["Q5"]
    assert [q.id for q in s.all({"subject": ["biology", "chemistry"]})] == ["B1"]
    assert "brand_new_key" in s.tag_keys()
