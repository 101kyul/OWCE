"""Inspect Task wrapper carrying trajectory metadata (parent, depth,
weakness tag) and the result type it produces.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog
from inspect_ai.scorer import choice, includes
from inspect_ai.solver import multiple_choice

MAX_CONNECTIONS = 10   # parallel model connections per Inspect eval
TIME_LIMIT_S = 600     # hard wall-clock limit per eval, so a hung call cannot stall a run
FAIL_ON_ERROR = 0.2    # the eval fails only if more than 20% of its samples error
RETRY_ON_ERROR = 2     # each errored sample is retried this many times first


@dataclass
class TaskResult:
    """Result of executing a wrapped Inspect task."""

    task_id: str
    score: float
    scores: dict[str, float]
    metadata: dict[str, Any]
    parent_id: Optional[str]
    depth: int
    samples: list[dict[str, Any]] = field(default_factory=list)
    threshold: float = 0.5

    @property
    def passed(self) -> bool:
        """Did the task pass (score at or above the threshold)?"""
        return self.score >= self.threshold

    @classmethod
    def from_eval_log(
        cls,
        eval_log: EvalLog,
        task_id: str,
        parent_id: Optional[str],
        depth: int,
        threshold: float = 0.5,
    ) -> "TaskResult":
        # Inspect's eval() returns status="error" instead of raising; a failed
        # eval must not be read as a score.
        status = getattr(eval_log, "status", "success")
        if status != "success":
            err = getattr(eval_log, "error", None)
            raise RuntimeError(
                f"inspect eval ended with status={status!r} for task "
                f"{task_id}: {getattr(err, 'message', err)}")

        # Primary score: the first scorer's metric, preferring common metric names
        primary_score = 0.0
        if eval_log.results and eval_log.results.scores:
            first_scorer = eval_log.results.scores[0]
            metric_names_to_try = ["accuracy", "mean", "score", "all"]

            for metric_name in metric_names_to_try:
                if metric_name in first_scorer.metrics:
                    primary_score = first_scorer.metrics[metric_name].value
                    break
            else:
                if first_scorer.metrics:
                    first_metric = next(iter(first_scorer.metrics.values()))
                    primary_score = first_metric.value

        scores = {}
        if eval_log.results and eval_log.results.scores:
            for score in eval_log.results.scores:
                for metric_name, metric in score.metrics.items():
                    scores[metric_name] = metric.value

        samples = []
        if eval_log.samples:
            for sample in eval_log.samples:
                # sample.scores is a dict keyed by scorer name; take the first
                score_value = None
                if sample.scores:
                    first_scorer = next(iter(sample.scores.values()))
                    score_value = first_scorer.value if hasattr(first_scorer, 'value') else first_scorer

                # sample.output is usually a ModelOutput; keep only its completion text
                output_text = sample.output
                if hasattr(sample.output, 'completion'):
                    output_text = sample.output.completion
                elif not isinstance(sample.output, str):
                    output_text = str(sample.output)

                sample_dict = {
                    "id": sample.id,
                    "input": sample.input if isinstance(sample.input, str) else str(sample.input),
                    "target": sample.target,
                    "output": output_text,
                    "score": score_value,
                    "metadata": sample.metadata if hasattr(sample, 'metadata') else {},
                }
                samples.append(sample_dict)

        return cls(
            task_id=task_id,
            score=primary_score,
            scores=scores,
            metadata={},
            parent_id=parent_id,
            depth=depth,
            samples=samples,
            threshold=threshold,
        )


@dataclass
class EvalTask:
    """Inspect Task plus trajectory metadata."""

    task: Task
    parent_id: Optional[str] = None
    depth: int = 0
    weakness_tag: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def execute(
        self,
        model: str,
        limit: Optional[int] = None,
        threshold: float = 0.5,
    ) -> TaskResult:
        """Run the task on ``model`` with Inspect eval; ``limit`` caps the
        number of samples and ``threshold`` is the pass mark recorded on the
        result.
        """
        eval_logs = inspect_eval(
            tasks=self.task,
            model=model,
            limit=limit,
            max_connections=MAX_CONNECTIONS,
            log_level="warning",
            time_limit=TIME_LIMIT_S,
            fail_on_error=FAIL_ON_ERROR,
            retry_on_error=RETRY_ON_ERROR,
        )

        if not eval_logs:
            raise ValueError("Inspect eval returned no logs")

        eval_log = eval_logs[0]

        result = TaskResult.from_eval_log(
            eval_log=eval_log,
            task_id=self.id,
            parent_id=self.parent_id,
            depth=self.depth,
            threshold=threshold,
        )

        return result

    @classmethod
    def from_item(
        cls,
        item: dict[str, Any],
        scorer=None,
        parent_id: Optional[str] = None,
        depth: int = 0,
    ) -> "EvalTask":
        """Build an EvalTask from an item dict (question, answer or
        gold_answer, answer_type, optional choices and meta). With
        ``scorer=None`` the scorer is chosen by answer type: choice() for
        multiple choice, includes() otherwise.
        """
        question = item.get("question", "")
        target = item.get("answer") or item.get("gold_answer", "")
        choices = item.get("choices", None)
        answer_type = item.get("answer_type", "multiple_choice")

        clean_choices = None
        has_real_choices = False
        if choices and isinstance(choices, list):
            clean_choices = [str(c).strip() for c in choices]
            # Strip option labels ("A) text", "B: text", "(C) text") when every
            # choice carries its own letter in order
            labels = [re.match(r'^\(?([A-Ea-e])[:.)]\s*', c) for c in clean_choices]
            if all(labels) and all(m.group(1).upper() == "ABCDE"[i] for i, m in enumerate(labels)):
                clean_choices = [c[m.end():] or c for c, m in zip(clean_choices, labels)]
            has_real_choices = any(len(c) > 1 for c in clean_choices)

        # Choices that are only single letters carry no content; drop them
        if clean_choices and not has_real_choices:
            clean_choices = None

        # Normalise the multiple-choice target to a single upper-case letter
        is_mc = answer_type == "multiple_choice" and clean_choices
        if is_mc and target:
            target = str(target).strip()
            letter_match = re.match(r'^([A-Ea-e])[\s:.)]+', target)
            if letter_match:
                target = letter_match.group(1).upper()
            elif len(target) == 1 and target.upper() in "ABCDE":
                target = target.upper()

        solver = None
        if is_mc:
            # multiple_choice() presents the choices and extracts the letter;
            # choice() scores it
            sample = Sample(
                input=question,
                target=target,
                choices=clean_choices,
                id=item.get("id", ""),
                metadata=item.get("meta", {}),
            )
            solver = multiple_choice()
            if scorer is None:
                scorer = choice()
        else:
            # Non-MC: any choices are embedded in the question text and the
            # answer is scored with includes()
            if clean_choices:
                has_embedded = any(
                    marker in question
                    for marker in ["(A)", "(B)", "A)", "B)", "A.", "B."]
                )
                if not has_embedded:
                    labels = "ABCDE"
                    choices_text = "\n".join(
                        f"({labels[i]}) {c}" for i, c in enumerate(clean_choices)
                        if i < len(labels)
                    )
                    question = f"{question}\n\n{choices_text}"

            sample = Sample(
                input=question,
                target=str(target),
                id=item.get("id", ""),
                metadata=item.get("meta", {}),
            )
            if scorer is None:
                scorer = includes()

        task_kwargs_inspect = {"dataset": [sample], "scorer": scorer}
        if solver is not None:
            task_kwargs_inspect["solver"] = solver
        task = Task(**task_kwargs_inspect)

        # weakness_tag may live in meta, cluster_weakness or at the top level
        weakness_tag = (
            item.get("meta", {}).get("weakness_tag")
            or item.get("cluster_weakness")
            or item.get("weakness_tag")
        )

        kwargs = {
            "task": task,
            "parent_id": parent_id,
            "depth": depth,
            "weakness_tag": weakness_tag,
            "meta": item.get("meta", {}),
        }

        item_id = item.get("id", "")
        if item_id:
            kwargs["id"] = item_id

        return cls(**kwargs)
