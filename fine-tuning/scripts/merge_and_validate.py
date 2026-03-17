#!/usr/bin/env python3
"""Merge per-source JSONL files, validate, deduplicate, and produce train/eval split.

Usage:
    python merge_and_validate.py [--raw-dir ../data/raw] [--output-dir ../data] [--eval-fraction 0.1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_METADATA_FIELDS = [
    "source",
    "source_url",
    "source_section",
    "example_type",
    "topics",
    "generation_prompt_id",
    "generated_at",
    "generator_model",
]

VALID_EXAMPLE_TYPES = {"qa", "instruction", "passage"}

VALID_ROLES = {"system", "user", "assistant"}


def load_raw_examples(raw_dir: Path) -> list[dict]:
    """Read all *.jsonl from raw_dir, return list of parsed dicts."""
    examples: list[dict] = []
    jsonl_files = sorted(raw_dir.glob("*.jsonl"))
    for fpath in jsonl_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    obj["_source_file"] = fpath.name
                    obj["_source_line"] = line_no
                    examples.append(obj)
                except json.JSONDecodeError as exc:
                    # Store as an invalid example so the validation log can capture it.
                    examples.append(
                        {
                            "_source_file": fpath.name,
                            "_source_line": line_no,
                            "_parse_error": str(exc),
                        }
                    )
    return examples


def validate_example(example: dict) -> list[str]:
    """Validate a single example. Return list of error strings (empty = valid).

    Checks:
    - messages is a list with 3 elements (system, user, assistant)
    - Each message has role and content
    - Assistant content > 50 characters
    - metadata has required fields: source, source_url, source_section,
      example_type, topics, generation_prompt_id, generated_at, generator_model
    - example_type is one of: qa, instruction, passage
    """
    errors: list[str] = []

    # JSON parse errors from load_raw_examples
    if "_parse_error" in example:
        errors.append(f"JSON parse error: {example['_parse_error']}")
        return errors

    # --- messages validation ---
    messages = example.get("messages")
    if not isinstance(messages, list):
        errors.append("messages must be a list")
        return errors

    if len(messages) != 3:
        errors.append(f"messages must have exactly 3 elements, got {len(messages)}")
        return errors

    expected_roles = ["system", "user", "assistant"]
    for idx, expected_role in enumerate(expected_roles):
        msg = messages[idx]
        if not isinstance(msg, dict):
            errors.append(f"messages[{idx}] must be a dict")
            continue
        role = msg.get("role")
        if role != expected_role:
            errors.append(
                f"messages[{idx}] role must be '{expected_role}', got '{role}'"
            )
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            errors.append(f"messages[{idx}] content must be a non-empty string")

    # Assistant content length check
    if len(messages) == 3 and isinstance(messages[2], dict):
        assistant_content = messages[2].get("content", "")
        if isinstance(assistant_content, str) and len(assistant_content) <= 50:
            errors.append(
                f"assistant content must be > 50 characters, got {len(assistant_content)}"
            )

    # --- metadata validation ---
    metadata = example.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata must be a dict")
        return errors

    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            errors.append(f"metadata missing required field: {field}")

    example_type = metadata.get("example_type")
    if example_type is not None and example_type not in VALID_EXAMPLE_TYPES:
        errors.append(
            f"example_type must be one of {VALID_EXAMPLE_TYPES}, got '{example_type}'"
        )

    topics = metadata.get("topics")
    if topics is not None and not isinstance(topics, list):
        errors.append("metadata.topics must be a list")

    return errors


def _content_hash(example: dict) -> str:
    """Hash of user+assistant content for exact dedup."""
    messages = example.get("messages", [])
    user_content = messages[1].get("content", "") if len(messages) > 1 else ""
    assistant_content = messages[2].get("content", "") if len(messages) > 2 else ""
    combined = user_content + "\x00" + assistant_content
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def exact_dedup(examples: list[dict]) -> tuple[list[dict], int]:
    """Remove exact duplicates based on hash of user+assistant content.

    Returns (deduped, n_removed).
    """
    seen: set[str] = set()
    deduped: list[dict] = []
    n_removed = 0
    for ex in examples:
        h = _content_hash(ex)
        if h in seen:
            n_removed += 1
        else:
            seen.add(h)
            deduped.append(ex)
    return deduped, n_removed


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Word-level Jaccard similarity between two texts."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def fuzzy_dedup(
    examples: list[dict], threshold: float = 0.85
) -> tuple[list[dict], list[dict]]:
    """Flag near-duplicate examples. Returns (kept, flagged).

    Uses jaccard_similarity on user content. O(n^2) is fine for ~2000 examples.
    """
    flagged_indices: set[int] = set()

    # Extract user content for comparison
    user_contents: list[str] = []
    for ex in examples:
        messages = ex.get("messages", [])
        user_content = messages[1].get("content", "") if len(messages) > 1 else ""
        user_contents.append(user_content)

    for i in range(len(examples)):
        if i in flagged_indices:
            continue
        for j in range(i + 1, len(examples)):
            if j in flagged_indices:
                continue
            sim = jaccard_similarity(user_contents[i], user_contents[j])
            if sim >= threshold:
                flagged_indices.add(j)

    kept = [ex for idx, ex in enumerate(examples) if idx not in flagged_indices]
    flagged = [ex for idx, ex in enumerate(examples) if idx in flagged_indices]
    return kept, flagged


def compute_coverage_stats(examples: list[dict]) -> dict:
    """Compute stats: by_source, by_type, by_topic, total, avg_lengths."""
    by_source: dict[str, int] = defaultdict(int)
    by_type: dict[str, int] = defaultdict(int)
    by_topic: dict[str, int] = defaultdict(int)

    user_lengths: list[int] = []
    assistant_lengths: list[int] = []

    for ex in examples:
        metadata = ex.get("metadata", {})
        source = metadata.get("source", "unknown")
        example_type = metadata.get("example_type", "unknown")
        topics = metadata.get("topics", [])

        by_source[source] += 1
        by_type[example_type] += 1
        for topic in topics:
            by_topic[topic] += 1

        messages = ex.get("messages", [])
        if len(messages) > 1:
            user_lengths.append(len(messages[1].get("content", "")))
        if len(messages) > 2:
            assistant_lengths.append(len(messages[2].get("content", "")))

    return {
        "total": len(examples),
        "by_source": dict(sorted(by_source.items())),
        "by_type": dict(sorted(by_type.items())),
        "by_topic": dict(sorted(by_topic.items())),
        "avg_user_length": (
            sum(user_lengths) / len(user_lengths) if user_lengths else 0
        ),
        "avg_assistant_length": (
            sum(assistant_lengths) / len(assistant_lengths)
            if assistant_lengths
            else 0
        ),
    }


def format_coverage_report(stats: dict) -> str:
    """Format as markdown table."""
    lines: list[str] = []
    lines.append("# SFT Domain Knowledge — Coverage Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append("")
    lines.append(f"**Total examples**: {stats['total']}")
    lines.append(
        f"**Avg user prompt length**: {stats['avg_user_length']:.0f} chars"
    )
    lines.append(
        f"**Avg assistant response length**: {stats['avg_assistant_length']:.0f} chars"
    )
    lines.append("")

    # By source
    lines.append("## Examples by Source")
    lines.append("")
    lines.append("| Source | Count |")
    lines.append("|--------|------:|")
    for source, count in stats["by_source"].items():
        lines.append(f"| {source} | {count} |")
    lines.append("")

    # By type
    lines.append("## Examples by Type")
    lines.append("")
    lines.append("| Type | Count |")
    lines.append("|------|------:|")
    for etype, count in stats["by_type"].items():
        lines.append(f"| {etype} | {count} |")
    lines.append("")

    # By topic
    lines.append("## Examples by Topic")
    lines.append("")
    lines.append("| Topic | Count |")
    lines.append("|-------|------:|")
    for topic, count in stats["by_topic"].items():
        lines.append(f"| {topic} | {count} |")
    lines.append("")

    return "\n".join(lines)


def split_train_eval(
    examples: list[dict], eval_fraction: float = 0.1, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Stratified split by source. Returns (train, eval)."""
    # Group by source
    by_source: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        source = ex.get("metadata", {}).get("source", "unknown")
        by_source[source].append(ex)

    rng = random.Random(seed)
    train: list[dict] = []
    eval_set: list[dict] = []

    for source in sorted(by_source.keys()):
        group = by_source[source]
        rng.shuffle(group)
        n_eval = max(1, round(len(group) * eval_fraction))
        # If there's only 1 example, put it in train
        if len(group) == 1:
            train.extend(group)
        else:
            eval_set.extend(group[:n_eval])
            train.extend(group[n_eval:])

    return train, eval_set


def _write_jsonl(path: Path, examples: list[dict]) -> None:
    """Write examples to a JSONL file, stripping internal metadata."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            # Strip internal bookkeeping fields
            clean = {k: v for k, v in ex.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Merge, validate, deduplicate, and split SFT data."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
        help="Directory containing per-source JSONL files (default: ../data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Output directory for train/eval JSONL and reports (default: ../data)",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.1,
        help="Fraction of examples to hold out for evaluation (default: 0.1)",
    )
    args = parser.parse_args(argv)

    raw_dir: Path = args.raw_dir
    output_dir: Path = args.output_dir
    eval_fraction: float = args.eval_fraction

    if not raw_dir.is_dir():
        print(f"Error: raw directory does not exist: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    print(f"Loading examples from {raw_dir} ...")
    all_examples = load_raw_examples(raw_dir)
    print(f"  Loaded {len(all_examples)} raw examples")

    # 2. Validate
    valid: list[dict] = []
    validation_log: list[dict] = []
    for ex in all_examples:
        errors = validate_example(ex)
        if errors:
            validation_log.append(
                {
                    "source_file": ex.get("_source_file", "unknown"),
                    "source_line": ex.get("_source_line", -1),
                    "errors": errors,
                    "action": "dropped",
                }
            )
        else:
            valid.append(ex)
    print(f"  Valid: {len(valid)}, dropped: {len(validation_log)}")

    # 3. Exact dedup
    deduped, n_exact = exact_dedup(valid)
    print(f"  After exact dedup: {len(deduped)} (removed {n_exact})")

    # 4. Fuzzy dedup
    kept, flagged = fuzzy_dedup(deduped)
    for ex in flagged:
        validation_log.append(
            {
                "source_file": ex.get("_source_file", "unknown"),
                "source_line": ex.get("_source_line", -1),
                "errors": ["near-duplicate (fuzzy dedup)"],
                "action": "flagged_removed",
            }
        )
    print(f"  After fuzzy dedup: {len(kept)} (flagged {len(flagged)})")

    # 5. Coverage stats
    stats = compute_coverage_stats(kept)
    report = format_coverage_report(stats)

    # 6. Split
    train, eval_set = split_train_eval(kept, eval_fraction=eval_fraction)
    print(f"  Train: {len(train)}, Eval: {len(eval_set)}")

    # 7. Write outputs
    train_path = output_dir / "sft_domain_knowledge_train.jsonl"
    eval_path = output_dir / "sft_domain_knowledge_eval.jsonl"
    report_path = output_dir / "coverage_report.md"
    log_path = output_dir / "validation_log.jsonl"

    _write_jsonl(train_path, train)
    _write_jsonl(eval_path, eval_set)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    with open(log_path, "w", encoding="utf-8") as f:
        for entry in validation_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nOutputs written to {output_dir}:")
    print(f"  {train_path.name}  ({len(train)} examples)")
    print(f"  {eval_path.name}  ({len(eval_set)} examples)")
    print(f"  {report_path.name}")
    print(f"  {log_path.name}  ({len(validation_log)} entries)")


if __name__ == "__main__":
    main()
