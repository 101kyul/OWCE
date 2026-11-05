"""Weakness tree of Inspect tasks."""

from typing import Optional

from owce.tasks import EvalTask


class Trajectory:
    """Trajectory tree of Inspect tasks: expand failed tasks into follow-ups."""

    def __init__(
        self,
        seed_tasks: list[EvalTask],
        generator,   # any object with generate_followup(parent_task, failure_result, ...)
        threshold: float = 0.7,
    ):
        """``seed_tasks`` form depth 0. ``threshold`` is the score in [0, 1]
        below which a task counts as failed and spawns follow-ups.
        """
        self.nodes = {}
        self.results = {}
        self.generator = generator
        self.threshold = threshold

        for task in seed_tasks:
            self.add_node(task)

    def add_node(self, task: EvalTask):
        self.nodes[task.id] = task

    def expand(
        self,
        target_model: str,
        max_depth: int = 3,
        limit_per_task: Optional[int] = None,
        max_candidates: int = 2,
    ):
        """Breadth-first expansion: run every task at the current depth on
        ``target_model`` and generate up to ``max_candidates`` follow-ups for
        each task scoring below the threshold, down to ``max_depth``.
        ``limit_per_task`` caps samples for depth-0 tasks only.
        """
        current_depth = 0

        while current_depth <= max_depth:
            tasks_at_depth = [
                t for t in self.nodes.values() if t.depth == current_depth
            ]

            if not tasks_at_depth:
                break

            print(f"\nDepth {current_depth}: {len(tasks_at_depth)} tasks")

            for task in tasks_at_depth:
                if task.id in self.results:
                    # Pre-seeded result (e.g. an injected depth-0 result): skip
                    # execution but still generate follow-ups.
                    result = self.results[task.id]
                    print(f"  [pre-seeded] {task.id[:8]} score={result.score:.2f}")
                else:
                    print(f"  Executing {task.id[:8]}")

                    sample_limit = limit_per_task if current_depth == 0 else None

                    result = task.execute(
                        model=target_model,
                        limit=sample_limit,
                        threshold=self.threshold,
                    )

                    self.results[task.id] = result

                    print(f"    Score: {result.score:.2f} ({'pass' if result.passed else 'fail'})")

                if result.score < self.threshold and current_depth < max_depth:
                    print("    Generating follow-ups")
                    followups = self.generator.generate_followup(
                        parent_task=task,
                        failure_result=result,
                        max_candidates=max_candidates,
                    )

                    print(f"      Generated {len(followups)} followup tasks")
                    for fu in followups:
                        self.add_node(fu)
                        print(f"      + {fu.id[:8]} (depth={fu.depth})")

            current_depth += 1

        print(f"\nExpansion complete: {len(self.nodes)} nodes, {len(self.results)} results")
