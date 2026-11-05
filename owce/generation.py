"""Cluster-based batch generation.

Failed items that share a weakness tag are presented together in one prompt,
so the generator sees several examples of the same failure pattern and the
number of API calls is per cluster rather than per item. Candidates carry a
generator-reported difficulty score that is checked against the parent.
"""

from typing import List, Dict, Any, Optional
import json
import math
import re

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```")


def extract_json_array(text: str) -> Optional[str]:
    """Return the JSON array in a model reply: a fenced ```json block if present,
    otherwise the span from the first '[' to the last ']'."""
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1)
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def generate_cluster_batch(
    *,
    runner,
    model_cfg: dict,
    cluster_items: List[Dict[str, Any]],
    shared_weakness_tag: str,
    max_candidates: int = 3,
    depth: int,
    min_difficulty_increase: float = 1.5,
    max_display_items: int = 5,
) -> List[Dict[str, Any]]:
    """Generate up to ``max_candidates`` questions from a cluster of failed
    items sharing ``shared_weakness_tag``. At most ``max_display_items``
    examples are shown in the prompt. Candidates whose self-reported
    difficulty is below parent difficulty + ``min_difficulty_increase`` are
    dropped.
    """

    print(f"      [cluster-batch] Processing cluster of {len(cluster_items)} items")
    print(f"      [cluster-batch] Shared weakness: {shared_weakness_tag}")

    # Depth-based heuristic for parent difficulty: base 5, +1 per level
    parent_difficulty = 5.0 + (depth * 1.0)
    required_difficulty = parent_difficulty + min_difficulty_increase

    prompt = _build_cluster_batch_prompt(
        cluster_items=cluster_items,
        shared_weakness_tag=shared_weakness_tag,
        parent_difficulty=parent_difficulty,
        required_difficulty=required_difficulty,
        max_candidates=max_candidates,
        max_display_items=max_display_items,
    )

    raw = runner.generate(model_cfg, [prompt])
    text = raw[0] if raw else ""

    payload = extract_json_array(text) or text
    try:
        data = json.loads(payload)
    except Exception:
        # Some generators emit backslash sequences that are not JSON escapes
        # (e.g. "\2" in chemistry notation). Escape lone backslashes and
        # try once more before giving up on the reply.
        try:
            data = json.loads(_repair_json_escapes(payload))
        except Exception:
            print("      [cluster-batch] Failed to parse JSON")
            print(f"      [cluster-batch] Raw response (first 500): {repr(text[:500])}")
            return []

    if not isinstance(data, list):
        return []

    print(f"      [cluster-batch] Parsed {len(data)} raw candidates")

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
            print(f"      [cluster-batch] [{i}] rejected: missing or malformed difficulty score")
            continue

        if difficulty_score < required_difficulty:
            print(f"      [cluster-batch] [{i}] rejected: too easy ({difficulty_score:.1f} < {required_difficulty:.1f})")
            continue

        answer_type = str(item.get("answer_type", "")).strip()

        # The cluster's domain is the most common domain among its items
        domains = [str(ci.get("domain", "unknown")) for ci in cluster_items]
        domain = max(set(domains), key=domains.count) if domains else "unknown"

        verified_item = {
            "question": question,
            "answer_type": answer_type,
            "domain": domain,
            "gold_answer": str(item.get("gold_answer", "")).strip(),
            "difficulty_score": difficulty_score,
            "difficulty_justification": str(item.get("difficulty_justification", "")).strip(),
            "cluster_size": len(cluster_items),
            "cluster_weakness": shared_weakness_tag,
            # The seed items this cluster was built from, kept for provenance
            "source_item_ids": [ci["_id"] for ci in cluster_items if ci.get("_id")],
        }

        if answer_type == "multiple_choice":
            choices = item.get("choices")
            if not (isinstance(choices, list) and 3 <= len(choices) <= 5):
                print(f"      [cluster-batch] [{i}] rejected: multiple choice needs 3-5 choices")
                continue
            gold = normalize_choice_letter(verified_item["gold_answer"], len(choices))
            if gold is None:
                print(f"      [cluster-batch] [{i}] rejected: gold answer "
                      f"{verified_item['gold_answer']!r} is not one of the choices")
                continue
            verified_item["gold_answer"] = gold
            verified_item["choices"] = [str(c).strip() for c in choices]

        verified.append(verified_item)
        print(f"      [cluster-batch] [{i}] accepted (score: {difficulty_score:.1f})")

    print(f"      [cluster-batch] Final: {len(verified)}/{len(data)} verified")
    print(f"      [cluster-batch] {len(cluster_items)} items -> 1 API call")

    return verified


def _repair_json_escapes(payload: str) -> str:
    """Escape backslashes that do not start a valid JSON escape."""
    return re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', payload)


def normalize_choice_letter(gold: str, n_choices: int) -> Optional[str]:
    """The upper-case letter a multiple-choice key names ("b", "B)", "B: text"
    -> "B"), or None when it is not a letter within the first ``n_choices``."""
    m = re.match(r"^\(?([A-Ea-e])\)?(?:[\s:.)]|$)", str(gold).strip())
    if not m:
        return None
    letter = m.group(1).upper()
    return letter if letter in "ABCDE"[:n_choices] else None


def _build_cluster_batch_prompt(
    *,
    cluster_items: List[Dict[str, Any]],
    shared_weakness_tag: str,
    parent_difficulty: float,
    required_difficulty: float,
    max_candidates: int,
    max_display_items: int,
) -> str:
    display_items = cluster_items[:max_display_items]

    examples_text = ""
    for i, item in enumerate(display_items, 1):
        question = item.get("question", "")
        gold = item.get("answer", item.get("gold_answer", ""))
        model_answer = item.get("output", item.get("target_answer", ""))

        examples_text += f"\n**Example {i}:**\n"
        examples_text += f"Question: {question}\n"
        examples_text += f"Correct Answer: {gold}\n"
        examples_text += f"Model's Wrong Answer: {model_answer}\n"

    if len(cluster_items) > max_display_items:
        examples_text += f"\n...(and {len(cluster_items) - max_display_items} more similar failures)\n"

    prompt = f"""IMPORTANT: Output ONLY a JSON array. Do NOT write any explanation, analysis, or preamble. Start your response directly with `[` and end with `]`.

You are an ADVERSARIAL QUESTION DESIGNER. Your goal is to craft questions that will BREAK the target AI model.

**INTELLIGENCE BRIEFING:**
The target model has a SYSTEMATIC FAILURE PATTERN. It failed on ALL {len(cluster_items)} questions below in the EXACT SAME WAY.
- Weakness type: {shared_weakness_tag}
- This is NOT random error — it reveals a STRUCTURAL BLIND SPOT in the model's reasoning.

**EVIDENCE (Model's Failures):**
{examples_text}

**YOUR MISSION:**
Generate {max_candidates} questions that:
1. EXPLOIT this exact blind spot — not just "hard" questions, but questions specifically designed to trigger THIS failure mode
2. Include DECEPTIVE ELEMENTS that look easy on the surface but contain the exact reasoning trap the model falls into
3. Design DISTRACTORS that match the model's observed error pattern (the wrong answers it tends to give)
4. Make the correct answer COUNTERINTUITIVE — require the model to resist its default reasoning heuristic

**ADVERSARIAL STRATEGIES (use at least 2 per question):**
- Cognitive trap: Setup that primes the wrong reasoning path, then requires a reversal
- Misleading surface features: Numbers/words that suggest an incorrect but plausible answer
- Multi-step dependency: Each step's correctness depends on the previous (error compounds)
- Edge case exploitation: Boundary conditions where the model's heuristic breaks down
- Distractor alignment: Wrong answer choices match the model's typical error pattern from the evidence above
- Negation/exception: Rules with exceptions that the model is likely to miss
- Scale mismatch: Problems where intuitive magnitude estimates are wrong

**DIFFICULTY: {required_difficulty:.1f}+ / 10** (parent was {parent_difficulty:.1f})

**OUTPUT FORMAT:**
```json
[
  {{
    "question": "Your adversarial question targeting the blind spot",
    "answer_type": "multiple_choice",
    "choices": ["option A text", "option B text", "option C text", "option D text"],
    "gold_answer": "B",
    "difficulty_score": 8.5,
    "difficulty_justification": "Exploits {shared_weakness_tag} by [specific mechanism]. Uses [adversarial strategy] to trigger the same failure mode seen in the evidence.",
    "adversarial_strategy": "cognitive_trap + distractor_alignment"
  }}
]
```

**RULES:**
- "choices" must contain the ACTUAL TEXT of each option, NOT just letters like "A","B","C","D"
- "gold_answer" must be a SINGLE LETTER (A, B, C, or D) indicating the correct choice
- difficulty_score must be >= {required_difficulty:.1f} — questions below this are REJECTED
- EVERY question must target weakness: {shared_weakness_tag}
- The correct answer must be VERIFIABLY correct — no ambiguity
"""

    return prompt


def _extract_difficulty_score(item: dict) -> float | None:
    """The generator's difficulty score of an item, from its difficulty_score
    field, a "<n>/10" in the justification, or a score field."""
    score = item.get("difficulty_score")
    if score is not None:
        try:
            return float(score)
        except (ValueError, TypeError):
            pass

    justification = str(item.get("difficulty_justification", ""))
    match = re.search(r'(\d+\.?\d*)/10', justification)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    score = item.get("score")
    if score is not None:
        try:
            return float(score)
        except (ValueError, TypeError):
            pass

    return None
