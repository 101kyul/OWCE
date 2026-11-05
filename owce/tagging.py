"""The twelve-tag weakness tagger: one pattern set per tag, matched against
the question text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


# The 12-tag weakness taxonomy, in five groups
WEAKNESS_KEYS: List[str] = [
    # A. Inferential
    "inference_chain",          # Multi-step reasoning / chain-of-thought failure
    "formal_logic",             # Boolean / deductive / symbolic logic error
    "causal_reasoning",         # Cause-effect / causal graph reasoning error
    # B. Computational
    "numeric_computation",      # Arithmetic / counting / unit calculation error
    "symbolic_algebra",         # Algebraic manipulation / equation solving error
    # C. Knowledge
    "domain_knowledge",         # Expert-level factual knowledge required
    "common_knowledge",         # General-world-knowledge gap
    # D. Context
    "demographic_stereotype",   # Applied demographic stereotype instead of context/unknown
    "negation_scope",           # Mishandled negation, exceptions, or 'NOT' framing
    "instruction_constraint",   # Violated explicit output constraint or format rule
    # E. Calibration and pragmatic
    "confidence_calibration",   # Overconfident wrong answer or unnecessary hedge
    "pragmatic_inference",      # Failed figurative/implied meaning
]

# Tags that match most questions; clustering uses a more specific tag when
# one matched.
GENERIC_TAGS = {"inference_chain", "numeric_computation", "common_knowledge"}


@dataclass
class WeaknessAnalysis:
    error_tags: List[str] = field(default_factory=list)


# Content regex patterns, one per tag

_INFERENCE_CHAIN = re.compile(
    r"(?:therefore|because|since|implies|conclude|infer|deduc|"
    r"if.*then|follows that|step.by.step|first.*then|finally|"
    r"multi.?step|chain|sequence of|in order to|first.*next|"
    r"reasoning|derive|proof|establish|result in|leads? to)",
    re.IGNORECASE,
)
_FORMAL_LOGIC = re.compile(
    r"(?:boolean|tautolog|syllogism|logical.*valid|deduction|"
    r"all.*are.*some|if.*then.*therefore|not.*and.*or|"
    r"dyck|parenthes|bracket.*match|formal.*fallac|necessarily|"
    r"truth table|entail|contraposit|biconditional|iff\b)",
    re.IGNORECASE,
)
_CAUSAL_REASONING = re.compile(
    r"(?:cause|effect\b|result.*in|lead.*to|because of|due to|"
    r"reason.*for|explain.*why|consequence|impact.*of|"
    r"what.*cause|why.*did|causal|outcome|trigger)",
    re.IGNORECASE,
)
_NUMERIC_COMPUTATION = re.compile(
    r"(?:calcul|comput|arithmeti|sum\b|total\b|add|subtract|multiply|divide|"
    r"percent|ratio|fraction|\d+\s*[\+\-\*\/\^]\s*\d+|"
    r"how many|how much|count\b|number of|price|cost|speed|distance|"
    r"how long|how far|probability|average|mean\b|median)",
    re.IGNORECASE,
)
_SYMBOLIC_ALGEBRA = re.compile(
    r"(?:simplif|evaluat.*express|variabl|polynom|algebra|"
    r"equation.*solv|factor\b|expand\b|derivative|integral\b|"
    r"matrix|vector\b|function.*of.*x|f\s*\(|√|∑|∫|"
    r"exponent|logarithm|quadratic|root of)",
    re.IGNORECASE,
)
_DOMAIN_KNOWLEDGE = re.compile(
    r"(?:scien|physic|chemi|biolog|medic|histor|geograph|"
    r"expert|specialist|technical|field of|discipline|"
    r"according to.*theory|law of|principle of|"
    r"phd|graduate|advanced|university|research|mechanism|"
    r"pathway|enzyme|reaction|element|compound|species)",
    re.IGNORECASE,
)
_COMMON_KNOWLEDGE = re.compile(
    r"(?:capital of|president of|author of|inventor of|discover|"
    r"famous for|known for|year.*born|year.*died|country|language|"
    r"general knowledge|popular culture|movie|sport|music|celebrity|"
    r"who.*was|what.*is the name|which.*country|when.*was)",
    re.IGNORECASE,
)
_DEMOGRAPHIC_STEREOTYPE = re.compile(
    r"(?:grandson|granddaughter|grandfather|grandmother|"
    r"older\s+(?:man|woman|person|adult)|younger\s+(?:man|woman|person|adult)|"
    r"elderly|senior\s+(?:man|woman|citizen)|"
    r"(?:Black|White|Asian|Hispanic|Latino|Latina|Arab|Indian|Native)\s+"
    r"(?:man|woman|person|student|worker|individual)|"
    r"(?:gay|lesbian|bisexual|transgender|LGBT\w*)|"
    r"(?:Muslim|Jewish|Christian|Hindu|Buddhist)\s+(?:man|woman|person)|"
    r"(?:disabled|wheelchair|visually.impaired|hearing.impaired|blind\s+(?:man|woman|person))|"
    r"(?:low.income|high.income|wealthy|working.class)\s+(?:man|woman|person|family))",
    re.IGNORECASE,
)

_NEGATION_SCOPE = re.compile(
    r"(?:\bnot\b|n't\b|never\b|neither\b|\bnor\b|"
    r"\bexcept\b|other than|"
    r"false.*statement|incorrect.*statement|"
    r"which.*\bnot\b|does not|is not|are not|"
    r"least likely|cannot|impossible|"
    r"EXCEPT|NOT correct|NOT true|NOT false|"
    r"which.*incorrect|which.*wrong|none of the)",
    re.IGNORECASE,
)
_INSTRUCTION_CONSTRAINT = re.compile(
    r"(?:\bmust\b|should\b|exactly\b|at least|at most|no more than|"
    r"constraint|condition|require|ensure|"
    r"in the format|express as|output as|write.*as|"
    r"round.*to|decimal place|significant fig|"
    r"given that|provided that|if and only if|such that|"
    r"do not|without using|only use)",
    re.IGNORECASE,
)
_CONFIDENCE_CALIBRATION = re.compile(
    r"(?:edge case|\bzero\b|negative\b|empty\b|null\b|\bnone\b|"
    r"minimum|maximum|extreme|infinity|undefined|"
    r"cannot determine|not enough information|insufficient|"
    r"uncertain|ambiguous|unclear|unknown|"
    r"how confident|probability|likelihood|"
    r"is it possible|could be|might be)",
    re.IGNORECASE,
)
_PRAGMATIC_INFERENCE = re.compile(
    r"(?:implicature|sarcasm|irony|humor|joke|"
    r"figurative|metaphor|idiom|colloquial|slang|"
    r"intended meaning|what.*really mean|"
    r"context.*suggest|between the lines|"
    r"ambiguous.*meaning|disambiguat|which.*meaning|"
    r"snark|double meaning|wordplay|pun)",
    re.IGNORECASE,
)

# (pattern, tag, confidence), highest priority first. The confidence only
# orders the tags; the first one is the primary tag.
_CONTENT_PATTERN_MAP: List[tuple[re.Pattern, str, float]] = [
    (_DEMOGRAPHIC_STEREOTYPE, "demographic_stereotype", 0.92),
    (_FORMAL_LOGIC,           "formal_logic",           0.85),
    (_NUMERIC_COMPUTATION,    "numeric_computation",    0.85),
    (_SYMBOLIC_ALGEBRA,       "symbolic_algebra",       0.80),
    (_CAUSAL_REASONING,       "causal_reasoning",       0.80),
    (_NEGATION_SCOPE,         "negation_scope",         0.80),
    (_PRAGMATIC_INFERENCE,    "pragmatic_inference",    0.78),
    (_INSTRUCTION_CONSTRAINT, "instruction_constraint", 0.78),
    (_INFERENCE_CHAIN,        "inference_chain",        0.75),
    (_DOMAIN_KNOWLEDGE,       "domain_knowledge",       0.72),
    (_CONFIDENCE_CALIBRATION, "confidence_calibration", 0.70),
    (_COMMON_KNOWLEDGE,       "common_knowledge",       0.65),
]


def _detect_content_tags(question: str) -> List[tuple[str, float]]:
    """Regex-based content detection. Returns (tag, confidence) sorted desc."""
    detected: List[tuple[str, float]] = []
    seen: set[str] = set()
    for pat, tag, conf in _CONTENT_PATTERN_MAP:
        if tag not in seen and pat.search(question):
            detected.append((tag, conf))
            seen.add(tag)
    if len(question) > 250 or question.count("?") > 1:
        if "inference_chain" not in seen:
            detected.append(("inference_chain", 0.60))
    return sorted(detected, key=lambda x: -x[1])


def analyze(question: str) -> WeaknessAnalysis:
    """Detect weakness tags for a failed item from its question text. A
    question that matches no pattern is tagged inference_chain."""
    tags = [tag for tag, _ in _detect_content_tags(question)]
    if not tags:
        tags = ["inference_chain"]
    return WeaknessAnalysis(error_tags=tags)

