"""Grouping failed items by their primary weakness tag."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


def cluster_failed_items(
    failed_items: List[Dict[str, Any]],
    min_cluster_size: int = 2,
) -> Dict[str, List[Dict[str, Any]]]:
    """Group failed items by their primary weakness tag.

    Items with tag "none" are sub-grouped by domain to avoid one giant
    undifferentiated cluster.  Clusters smaller than *min_cluster_size*
    are collected under the ``"_singleton"`` key for per-item fallback.
    """
    by_tag: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for item in failed_items:
        tags = list((item.get("weakness") or {}).get("error_tags") or [])
        primary = tags[0] if tags else "none"

        if primary == "none":
            domain = str(item.get("domain") or "unknown")
            primary = f"none:{domain}"

        by_tag[primary].append(item)

    clusters: Dict[str, List[Dict[str, Any]]] = {}
    singletons: List[Dict[str, Any]] = []

    for tag, items in by_tag.items():
        if len(items) >= min_cluster_size:
            clusters[tag] = items
        else:
            singletons.extend(items)

    if singletons:
        clusters["_singleton"] = singletons

    return clusters
