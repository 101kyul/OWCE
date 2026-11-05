"""Cluster generators: failed seed items are grouped and each group is sent to
the generator in one call.

ClusterGenerator holds the generator configuration and turns generated items
into tasks; the generators in experiments/ subclass it. WeaknessClusterGenerator
is the pipeline's generator: it tags every failure with one of the twelve
weakness categories (owce/tagging.py), groups by tag, and splits large groups
into sub-clusters of DISPLAY_SIZE failures.
"""

import uuid
from typing import List, Optional
from dataclasses import dataclass

from owce.tasks import EvalTask
from owce.clustering import cluster_failed_items
from owce.generation import generate_cluster_batch
from owce.generation_single import generate_with_difficulty_awareness
from owce.tagging import analyze as analyze_weakness, GENERIC_TAGS
from owce.models import generate_with_inspect


class Runner:
    """The generator model behind the generate(cfg, prompts) call the
    generation functions make."""

    def __init__(self, model_name):
        self.model_name = model_name

    def generate(self, cfg, prompts):
        return generate_with_inspect(self.model_name, cfg, prompts)


def failed_samples(failure_result) -> list:
    """The samples of a result that count as failures: Inspect's 'I' and 'P'
    grades, or a numeric score below 1."""
    failed = []
    for s in failure_result.samples:
        score = s.get("score", 0)
        if isinstance(score, str):
            if score in ['I', 'P']:
                failed.append(s)
        elif isinstance(score, (int, float)):
            if score < 1.0:
                failed.append(s)
    return failed


def output_text(sample) -> str:
    output_text = sample.get("output", "")
    if hasattr(output_text, 'completion'):
        output_text = output_text.completion
    elif not isinstance(output_text, str):
        output_text = str(output_text)
    return output_text


@dataclass
class ClusterGenerator:
    """Generator configuration shared by the cluster generators; subclasses
    implement generate_followup(parent_task, failure_result, max_candidates)."""

    model_name: str
    model_cfg: dict
    min_cluster_size: int = 2
    min_difficulty_increase: float = 1.5

    def _item_to_task(self, item: dict, parent_task: EvalTask) -> Optional[EvalTask]:
        try:
            if not item.get("id"):
                item["id"] = str(uuid.uuid4())[:8]

            # from_item reads weakness_tag from the item
            if not item.get("cluster_weakness") and not item.get("weakness_tag"):
                item["weakness_tag"] = parent_task.weakness_tag

            # Choices, seed provenance and the generator's difficulty score go
            # into meta, which the run file records
            meta = item.get("meta", {})
            if item.get("source_item_ids"):
                meta["source_item_ids"] = item["source_item_ids"]
            if item.get("choices"):
                meta["choices"] = item["choices"]
            if item.get("difficulty_score") is not None:
                meta["difficulty_score"] = item["difficulty_score"]
            item["meta"] = meta

            # scorer=None lets from_item pick a scorer by answer_type; the
            # parent's scorer may not fit (e.g. GSM8K's numeric scorer for a
            # generated multiple-choice question).
            task = EvalTask.from_item(
                item=item,
                scorer=None,
                parent_id=parent_task.id,
                depth=parent_task.depth + 1,
            )

            if not task.weakness_tag:
                task.weakness_tag = (
                    item.get("cluster_weakness")
                    or parent_task.weakness_tag
                )

            return task

        except Exception as e:
            print(f"      [cluster] Error creating task: {e}")
            return None


DISPLAY_SIZE = 4  # max failure examples shown to generator per probe call


@dataclass
class WeaknessClusterGenerator(ClusterGenerator):
    """The pipeline's generator: tags every failure, groups by tag, and
    splits large groups into DISPLAY_SIZE-item sub-clusters."""

    def generate_followup(
        self,
        parent_task: EvalTask,
        failure_result,
        max_candidates: int = 3,
    ) -> List[EvalTask]:

        print(f"      [weakness-cluster] Processing {parent_task.id[:8]}")

        failed = failed_samples(failure_result)
        if not failed:
            print("      [weakness-cluster] No failed samples")
            return []

        print(f"      [weakness-cluster] Found {len(failed)} failed samples")

        runner = Runner(self.model_name)

        cluster_items = []
        tag_counts = {}

        for sample in failed:
            question = sample.get("input", "")

            # The tag is a function of the question text only
            wa = analyze_weakness(question)
            raw_tags = list(wa.error_tags) if wa.error_tags else [parent_task.weakness_tag or "unknown"]

            # Specific tags take precedence over the generic ones for clustering
            specific_tags = [t for t in raw_tags if t not in GENERIC_TAGS]
            tags = specific_tags if specific_tags else raw_tags

            for t in tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1

            cluster_items.append({
                "question": question,
                "answer": sample.get("target", ""),
                "output": output_text(sample),
                "domain": parent_task.meta.get("domain", "unknown"),
                "answer_type": parent_task.meta.get("answer_type", "text"),
                "weakness": {"error_tags": tags},
                "_id": sample.get("id", ""),
            })

        print("      [weakness-cluster] Tag distribution:")
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
            print(f"        {tag}: {count}")

        clusters = cluster_failed_items(
            cluster_items,
            min_cluster_size=self.min_cluster_size,
        )

        print(f"      [weakness-cluster] Formed {len(clusters)} clusters")
        for tag, items in clusters.items():
            print(f"        {tag}: {len(items)} items -> "
                  f"{(len(items) + DISPLAY_SIZE - 1) // DISPLAY_SIZE} sub-clusters")

        all_followups = []
        remaining_budget = max_candidates

        # Mid-size clusters (3-15 items) go first: specific enough to target a
        # real weakness, with enough context for the generator. Then the
        # large ones, then the two-item ones, then the singletons; within a
        # tier the larger cluster first. Clusters left when the budget runs
        # out generate nothing.
        def cluster_priority(tag_items):
            tag, items = tag_items
            n = len(items)
            if 3 <= n <= 15:
                return 0
            elif n > 15:
                return 1
            else:
                return 2

        sorted_clusters = sorted(
            [(tag, items) for tag, items in clusters.items() if tag != "_singleton"],
            key=lambda x: (cluster_priority(x), -len(x[1])),
        )
        if "_singleton" in clusters:
            sorted_clusters.append(("_singleton", clusters["_singleton"]))

        for tag, items in sorted_clusters:
            if remaining_budget <= 0:
                print(f"      [weakness-cluster] Budget exhausted, skipping '{tag}'")
                break

            if tag == "_singleton":
                for item in items[:remaining_budget]:
                    item_tags = item.get("weakness", {}).get("error_tags", ["unknown"])
                    single_items = generate_with_difficulty_awareness(
                        runner=runner,
                        model_cfg=self.model_cfg,
                        parent=item,
                        target_answer=item.get("output", ""),
                        gold_answer=item.get("answer", ""),
                        weakness_tags=item_tags,
                        max_candidates=1,
                        depth=parent_task.depth + 1,
                        min_difficulty_increase=self.min_difficulty_increase,
                    )
                    for generated in single_items:
                        generated["weakness_tag"] = item_tags[0] if item_tags else "unknown"
                        task = self._item_to_task(generated, parent_task)
                        if task:
                            task.weakness_tag = item_tags[0] if item_tags else parent_task.weakness_tag
                            all_followups.append(task)
                            remaining_budget -= 1
            else:
                if len(items) > DISPLAY_SIZE:
                    sub_clusters = [
                        items[i:i + DISPLAY_SIZE]
                        for i in range(0, len(items), DISPLAY_SIZE)
                    ]
                else:
                    sub_clusters = [items]

                print(f"      [weakness-cluster] '{tag}': {len(items)} items -> "
                      f"{len(sub_clusters)} sub-clusters x {DISPLAY_SIZE}")

                for sc_idx, sub_items in enumerate(sub_clusters):
                    if remaining_budget <= 0:
                        break

                    print(f"        sub-cluster {sc_idx+1}/{len(sub_clusters)} "
                          f"({len(sub_items)} items, budget={remaining_budget})")

                    generated_items = generate_cluster_batch(
                        runner=runner,
                        model_cfg=self.model_cfg,
                        cluster_items=sub_items,
                        shared_weakness_tag=tag,
                        max_candidates=1,
                        depth=parent_task.depth + 1,
                        min_difficulty_increase=self.min_difficulty_increase,
                    )

                    for item in generated_items:
                        item["weakness_tag"] = tag
                        task = self._item_to_task(item, parent_task)
                        if task:
                            task.weakness_tag = tag
                            all_followups.append(task)
                            remaining_budget -= 1

        print(f"      [weakness-cluster] Generated {len(all_followups)} total followup tasks")
        return all_followups
