"""Difficulty-aware generator.

The generator is asked for an explicit 1-10 difficulty score per item, and
candidates scoring below the parent's estimated difficulty plus a margin are
rejected.
"""

from typing import List, Dict, Any
import json
import math

from owce.generation import (
    _extract_difficulty_score,
    _repair_json_escapes,
    extract_json_array,
    normalize_choice_letter,
)


def generate_with_difficulty_awareness(
    *,
    runner,
    model_cfg: dict,
    parent: Dict[str, Any],
    target_answer: str,
    gold_answer: str,
    weakness_tags: List[str],
    max_candidates: int = 3,
    depth: int,
    min_difficulty_increase: float = 1.5,
) -> List[Dict[str, Any]]:
    """Generate up to ``max_candidates`` follow-ups to ``parent`` with
    generator-reported difficulty scores. Candidates scoring below the
    parent's estimated difficulty + ``min_difficulty_increase`` are dropped.
    """

    print("      [single] Generating with difficulty scoring")

    # Depth-based heuristic for parent difficulty: base 5, +1 per level
    parent_difficulty = 5.0 + (depth * 1.0)
    required_difficulty = parent_difficulty + min_difficulty_increase

    print(f"      [single] Parent difficulty: {parent_difficulty:.1f}, Required: {required_difficulty:.1f}")

    prompt = _build_difficulty_prompt(
        parent=parent,
        target_answer=target_answer,
        gold_answer=gold_answer,
        weakness_tags=weakness_tags,
        parent_difficulty=parent_difficulty,
        required_difficulty=required_difficulty,
        max_candidates=max_candidates,
    )

    raw = runner.generate(model_cfg, [prompt])
    text = raw[0] if raw else ""

    payload = extract_json_array(text) or text
    try:
        data = json.loads(payload)
    except Exception:
        try:
            data = json.loads(_repair_json_escapes(payload))
        except Exception:
            print("      [single] Failed to parse JSON")
            return []

    if not isinstance(data, list):
        return []

    print(f"      [single] Parsed {len(data)} raw candidates")

    verified = []

    for i, item in enumerate(data, 1):
        if len(verified) >= max_candidates:
            break

        if not isinstance(item, dict):
            continue

        question = str(item.get("question", "")).strip()
        if not question:
            continue

        difficulty_score = _extract_difficulty_score(item)

        if difficulty_score is None or not math.isfinite(difficulty_score) or not 1 <= difficulty_score <= 10:
            print(f"      [single] [{i}] rejected: missing or malformed difficulty score")
            continue

        if difficulty_score < required_difficulty:
            print(f"      [single] [{i}] rejected: too easy (score: {difficulty_score:.1f} < {required_difficulty:.1f})")
            continue

        answer_type = str(item.get("answer_type", "")).strip()
        verified_item = {
            "question": question,
            "answer_type": answer_type,
            "domain": parent.get("domain", "unknown"),
            "gold_answer": str(item.get("gold_answer", "")).strip(),
            "difficulty_score": difficulty_score,
            "difficulty_justification": str(item.get("difficulty_justification", "")).strip(),
        }

        if answer_type == "multiple_choice":
            choices = item.get("choices")
            if not (isinstance(choices, list) and 3 <= len(choices) <= 5):
                print(f"      [single] [{i}] rejected: multiple choice needs 3-5 choices")
                continue
            gold = normalize_choice_letter(verified_item["gold_answer"], len(choices))
            if gold is None:
                print(f"      [single] [{i}] rejected: gold answer "
                      f"{verified_item['gold_answer']!r} is not one of the choices")
                continue
            verified_item["gold_answer"] = gold
            verified_item["choices"] = [str(c).strip() for c in choices]

        verified.append(verified_item)
        print(f"      [single] [{i}] accepted (score: {difficulty_score:.1f})")

    print(f"      [single] Final: {len(verified)}/{len(data)} verified")

    return verified


def _build_difficulty_prompt(
    *,
    parent: Dict[str, Any],
    target_answer: str,
    gold_answer: str,
    weakness_tags: List[str],
    parent_difficulty: float,
    required_difficulty: float,
    max_candidates: int,
) -> str:
    parent_q = parent.get("question", "")
    domain = parent.get("domain", "unknown")

    prompt = f"""You are an ADVERSARIAL QUESTION DESIGNER. Your goal is to craft a question that will BREAK the target AI model by exploiting its demonstrated weakness.

**FAILURE EVIDENCE:**
- Parent Question: {parent_q}
- Domain: {domain}
- Correct Answer: {gold_answer}
- Model's WRONG Answer: {target_answer}
- Identified Weakness: {', '.join(weakness_tags)}

The model got this wrong. Your job is to create {max_candidates} harder questions that exploit the SAME reasoning flaw.

**ADVERSARIAL STRATEGIES (use at least 2):**
- Cognitive trap: Setup that primes the wrong reasoning path, then requires a reversal
- Misleading surface features: Numbers/words that suggest an incorrect but plausible answer
- Multi-step dependency: Each step's correctness depends on the previous (error compounds)
- Edge case exploitation: Boundary conditions where the model's heuristic breaks down
- Distractor alignment: Wrong answer choices match the model's typical error pattern (it answered "{target_answer}" instead of "{gold_answer}")
- Negation/exception: Rules with exceptions that the model is likely to miss

**DIFFICULTY: {required_difficulty:.1f}+ / 10** (parent was {parent_difficulty:.1f})
Questions below {required_difficulty:.1f} will be REJECTED.

**OUTPUT FORMAT:**
```json
[
  {{
    "question": "Your adversarial follow-up question",
    "answer_type": "multiple_choice",
    "choices": ["option A text", "option B text", "option C text", "option D text"],
    "gold_answer": "B",
    "difficulty_score": 7.5,
    "difficulty_justification": "Exploits the same weakness by [mechanism]. Uses [adversarial strategy] to make the model repeat its error pattern.",
    "adversarial_strategy": "cognitive_trap + distractor_alignment"
  }}
]
```

**RULES:**
- "choices" must contain the ACTUAL TEXT of each option, NOT just letters
- "gold_answer" must be a SINGLE LETTER (A, B, C, or D)
- difficulty_score must be >= {required_difficulty:.1f}
- The correct answer must be VERIFIABLY correct — no ambiguity
- Design distractors to match the model's error pattern from the evidence above

Generate {max_candidates} adversarial questions, each with score >= {required_difficulty:.1f}.
"""

    return prompt

