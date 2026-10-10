"""Turn a natural-language instruction into a composition plan.

Two planners produce the same ``Plan`` structure:

* ``LLMPlanner`` — an LLM reads the instruction *and the catalog inventory*
  and returns JSON. It handles free phrasing ("a couple of kids next to some
  colourful shapes") but may only reference inventory names; anything else
  must be reported as missing.
* ``RulePlanner`` — deterministic parser (counts, colour adjectives, nouns,
  plural categories, layout and background words). Used offline and as the
  fallback when the LLM output does not validate.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from ..detection.colors import PROTOTYPES
from ..llm import chat_json, llm_available, DEFAULT_MODEL

COLOR_WORDS = [c for c in PROTOTYPES if c != "skin"]
NUMBERS = {"a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "pair": 2, "couple": 2, "three": 3,
           "four": 4, "five": 5, "six": 6, "some": 0, "several": 0}
CATEGORY_WORDS = {"shape": "shape", "fruit": "fruit", "person": "person", "people": "person",
                  "kid": "person", "child": "person", "children": "person", "swatch": "swatch",
                  "color swatch": "swatch"}
LAYOUTS = {"overlap": ["overlap", "overlapping", "stacked", "layered", "on top of each other"],
           "grid": ["grid"], "row": ["row", "side by side", "beside", "next to", "in a line"]}


@dataclass
class PlanItem:
    noun: str                    # object name ("apple") or category ("shape")
    count: int = 1               # 0 = "all distinct instances available"
    color: str | None = None
    is_category: bool = False


@dataclass
class Plan:
    items: list[PlanItem]
    layout: str = "row"
    background: str = "white"
    planner: str = "rules"
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def singular(word: str) -> str:
    irregular = {"people": "person", "children": "child", "boxes": "box", "dresses": "dress",
                 "watches": "watch", "peaches": "peach"}
    if word in irregular:
        return irregular[word]
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


class RulePlanner:
    PREFIX = re.compile(r"^(please\s+)?(create|generate|make|compose|build|draw|produce|render)?\s*"
                        r"(an?\s+)?(new\s+)?(composite\s+)?(image|picture|composition|scene|composite)?\s*"
                        r"(that\s+)?(containing|contains|with|of|showing|featuring|having|including)?\s*")

    def plan(self, instruction: str, inventory: dict | None = None) -> Plan:
        text = instruction.lower().strip().rstrip(".!")
        layout = next((k for k, words in LAYOUTS.items() if any(w in text for w in words)), "row")
        background = "white"
        m = re.search(r"\bon an? (\w+) background\b", text)
        if m:
            background = m.group(1)
            text = text.replace(m.group(0), " ")
        for words in LAYOUTS.values():
            for w in sorted(words, key=len, reverse=True):
                text = re.sub(rf"\b(arranged |placed |laid out )?(in an? |as an? )?{w}\b", ",", text)
        text = self.PREFIX.sub("", text, count=1)
        chunks = [c.strip() for c in re.split(r",|\band\b|&|\bplus\b|\balong with\b", text) if c.strip()]
        items = []
        for chunk in chunks:
            words = [w for w in re.findall(r"[a-z]+|\d+", chunk) if w not in ("of", "the", "some")]
            if not words:
                continue
            count, color, plural = 1, None, False
            if words[0].isdigit():
                count = int(words.pop(0))
            elif words[0] in NUMBERS:
                count = NUMBERS[words.pop(0)]
            colored = False
            rest = []
            for w in words:
                if w in COLOR_WORDS and color is None and w != words[-1]:
                    color = w
                elif w in ("colored", "coloured", "colorful", "colourful", "different", "various"):
                    colored = True
                else:
                    rest.append(w)
            if not rest:
                continue
            noun_raw = rest[-1]
            noun = singular(noun_raw)
            plural = noun != noun_raw
            phrase = " ".join(rest[-2:]) if len(rest) >= 2 else noun
            if singular(phrase) in CATEGORY_WORDS:
                noun = singular(phrase)
            is_cat = noun in CATEGORY_WORDS
            if is_cat:
                noun = CATEGORY_WORDS[noun]
            if (plural or colored) and count in (0, 1) and (is_cat or colored) and not words[0:1] == ["one"]:
                count = 0  # "colored shapes" → every distinct shape available
            items.append(PlanItem(noun, count, color, is_cat))
        return Plan(items, layout, background, "rules")


class LLMPlanner:
    SYSTEM = (
        "You convert an image-composition instruction into a JSON plan. The composition can ONLY use "
        "objects that exist in the catalog inventory provided; never invent assets. Output a single JSON "
        "object: {\"items\": [{\"noun\": str, \"count\": int, \"color\": str|null, \"is_category\": bool}], "
        "\"layout\": \"row\"|\"overlap\"|\"grid\", \"background\": str, \"missing\": [str]}. "
        "Rules: noun is an inventory object name (e.g. \"apple\") or, with is_category=true, an inventory "
        "category (e.g. \"shape\"). count=0 means 'all distinct instances of that category'. Put a colour "
        "only if the instruction requires that colour. If the instruction asks for something the "
        "inventory does not contain, still add it as an item AND list it in missing. Keep the order the "
        "instruction mentions things (left to right). layout is overlap only if overlapping/layering is "
        "requested; background defaults to white."
    )

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def plan(self, instruction: str, inventory: dict) -> Plan:
        inv_lines = [f"- {name} (category {v['category']}, colours {sorted(v['colors'])}, "
                     f"{v['count']} asset(s))" for name, v in sorted(inventory.items())]
        data = chat_json(self.SYSTEM, f"Inventory:\n" + "\n".join(inv_lines) +
                         f"\n\nInstruction: {instruction}", model=self.model)
        items = [PlanItem(str(i["noun"]).lower(), int(i.get("count", 1)), (i.get("color") or None),
                          bool(i.get("is_category", False))) for i in data["items"]]
        if not items:
            raise ValueError("empty plan")
        layout = data.get("layout", "row")
        if layout not in LAYOUTS:
            layout = "row"
        return Plan(items, layout, str(data.get("background") or "white"), f"llm:{self.model}",
                    [str(m) for m in data.get("missing", [])])


def make_plan(instruction: str, inventory: dict, use_llm: bool | None = None) -> tuple[Plan, list[str]]:
    """LLM plan when available and valid, otherwise the rule plan. Returns (plan, notes)."""
    notes = []
    if use_llm is None:
        use_llm = llm_available()
    if use_llm:
        try:
            return LLMPlanner().plan(instruction, inventory), notes
        except Exception as exc:
            notes.append(f"LLM planner failed ({type(exc).__name__}: {exc}); used rule planner")
    return RulePlanner().plan(instruction, inventory), notes
