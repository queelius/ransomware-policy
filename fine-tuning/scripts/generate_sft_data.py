#!/usr/bin/env python3
"""
SFT data generation orchestrator.

Reads sources from sources.yaml and prompt templates from prompts/*.yaml,
dispatches Claude Code via `claude -p` to generate training examples,
and writes per-source JSONL with full metadata.

Usage:
    python generate_sft_data.py [--sources ../data/sources.yaml] [--prompts ../prompts/] \
        [--output ../data/raw/] [--dry-run] [--source-key KEY]
"""

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SFTExample:
    messages: list[dict]
    metadata: dict


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_prompt_templates(prompts_dir: Path = None) -> list[dict]:
    """Load all prompt template YAML files from the prompts directory.

    Returns list of dicts, each with keys: source, description, templates.
    """
    if prompts_dir is None:
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"

    if not prompts_dir.is_dir():
        logger.warning("Prompts directory does not exist: %s", prompts_dir)
        return []

    results = []
    for yaml_path in sorted(prompts_dir.glob("*.yaml")):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data and "templates" in data:
            results.append(data)
        else:
            logger.warning("Skipping %s: no 'templates' key found", yaml_path.name)
    return results


def load_source_content(source_key: str, raw_content_dir: Path) -> list[dict]:
    """Load cached content for a source from raw_content/{source_key}/.

    Reads all files in the source directory and returns a list of
    {section: str, content: str} chunks. Each file becomes one chunk;
    the filename (without extension) is used as the section identifier.
    """
    source_dir = raw_content_dir / source_key
    if not source_dir.is_dir():
        logger.warning("No raw content directory for source '%s': %s", source_key, source_dir)
        return []

    chunks = []
    for file_path in sorted(source_dir.iterdir()):
        if file_path.is_file():
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning("Skipping non-UTF-8 file: %s", file_path)
                continue
            chunks.append({
                "section": file_path.stem,
                "content": text,
            })
    return chunks


def format_prompt(template: dict, content: str) -> str:
    """Substitute {content} placeholder in template prompt string."""
    prompt_text = template.get("prompt", "")
    return prompt_text.replace("{content}", content)


def invoke_claude(prompt: str, system: str) -> str:
    """Call claude -p with the given prompt and return raw response text.

    Uses subprocess with a 120-second timeout. Retries once on failure.
    """
    cmd = ["claude", "-p", prompt, "--no-input"]

    for attempt in range(2):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.warning(
                    "claude -p returned code %d (attempt %d/2): %s",
                    result.returncode,
                    attempt + 1,
                    result.stderr[:500],
                )
        except subprocess.TimeoutExpired:
            logger.warning("claude -p timed out (attempt %d/2)", attempt + 1)
        except FileNotFoundError:
            logger.error("'claude' command not found on PATH")
            return ""

    logger.error("claude -p failed after 2 attempts")
    return ""


def parse_generated_examples(raw_response: str, example_type: str) -> list[SFTExample]:
    """Parse Claude's JSON response into SFTExample objects.

    Handles three formats based on example_type:
      - qa:          [{"question": "...", "answer": "..."}]
      - instruction: [{"instruction": "...", "response": "..."}]
      - passage:     [{"passage": "..."}]

    Returns empty list on parse failure.
    """
    if not raw_response:
        return []

    # Try to extract JSON array from the response (may be wrapped in markdown)
    json_text = _extract_json_array(raw_response)
    if json_text is None:
        logger.warning("Could not extract JSON array from response")
        return []

    try:
        items = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error: %s", e)
        return []

    if not isinstance(items, list):
        logger.warning("Expected JSON array, got %s", type(items).__name__)
        return []

    examples = []
    for item in items:
        if not isinstance(item, dict):
            continue
        messages = _item_to_messages(item, example_type)
        if messages:
            examples.append(SFTExample(messages=messages, metadata={}))

    return examples


def _extract_json_array(text: str) -> Optional[str]:
    """Extract a JSON array from text that may contain markdown fences."""
    # Try to find a JSON array directly
    stripped = text.strip()

    # Remove markdown code fences if present
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        # Drop first line (```json or ```) and last line (```)
        inner_lines = []
        started = False
        for line in lines:
            if not started:
                started = True
                continue
            if line.strip() == "```":
                break
            inner_lines.append(line)
        stripped = "\n".join(inner_lines).strip()

    # Find the first [ and last ]
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]

    return None


def _item_to_messages(item: dict, example_type: str) -> list[dict]:
    """Convert a parsed item dict to a messages list based on type."""
    if example_type == "qa":
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        if question and answer:
            return [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
    elif example_type == "instruction":
        instruction = item.get("instruction", "").strip()
        response = item.get("response", "").strip()
        if instruction and response:
            return [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
    elif example_type == "passage":
        passage = item.get("passage", "").strip()
        if passage:
            return [
                {"role": "assistant", "content": passage},
            ]
    else:
        logger.warning("Unknown example_type: %s", example_type)
    return []


def build_metadata(
    source_key: str,
    source_url: str,
    source_section: str,
    example_type: str,
    topics: list[str],
    prompt_id: str,
    run_id: str,
) -> dict:
    """Build the metadata dict for an SFT example."""
    return {
        "source": source_key,
        "source_url": source_url,
        "source_section": source_section,
        "example_type": example_type,
        "topics": topics,
        "generation_prompt_id": prompt_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_model": "claude-opus-4-6",
        "quality_score": None,
        "run_id": run_id,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def load_sources(sources_path: Path) -> list[dict]:
    """Load sources.yaml and return the list of source dicts."""
    with open(sources_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("sources", [])


def _source_by_key(sources: list[dict], key: str) -> Optional[dict]:
    """Find a source dict by its key field."""
    for src in sources:
        if src.get("key") == key:
            return src
    return None


def generate_for_source(
    source: dict,
    templates: list[dict],
    raw_content_dir: Path,
    output_dir: Path,
    log_path: Path,
    run_id: str,
    dry_run: bool = False,
) -> int:
    """Generate SFT examples for a single source. Returns count of examples written."""
    source_key = source["key"]
    source_url = source.get("url", "")
    topics = source.get("topics", [])

    chunks = load_source_content(source_key, raw_content_dir)
    if not chunks:
        logger.info("No cached content for source '%s', skipping", source_key)
        return 0

    output_path = output_dir / f"{source_key}.jsonl"
    total_examples = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for template in templates:
            prompt_id = template["id"]
            example_type = template["type"]
            system_prompt = template.get("system", "")

            for chunk in chunks:
                section = chunk["section"]
                content = chunk["content"]
                formatted_prompt = format_prompt(template, content)

                if dry_run:
                    est_tokens = len(formatted_prompt.split()) + len(system_prompt.split())
                    print(f"[DRY RUN] source={source_key} section={section} "
                          f"prompt_id={prompt_id} est_tokens~{est_tokens}")
                    continue

                logger.info(
                    "Generating: source=%s section=%s prompt_id=%s",
                    source_key, section, prompt_id,
                )

                raw_response = invoke_claude(formatted_prompt, system_prompt)
                examples = parse_generated_examples(raw_response, example_type)

                for ex in examples:
                    # Prepend system message
                    if system_prompt:
                        ex.messages.insert(0, {"role": "system", "content": system_prompt})

                    ex.metadata = build_metadata(
                        source_key=source_key,
                        source_url=source_url,
                        source_section=section,
                        example_type=example_type,
                        topics=topics,
                        prompt_id=prompt_id,
                        run_id=run_id,
                    )

                    line = json.dumps({"messages": ex.messages, "metadata": ex.metadata}, ensure_ascii=False)
                    out_f.write(line + "\n")
                    total_examples += 1

                # Write generation log entry
                log_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "run_id": run_id,
                    "source_key": source_key,
                    "source_section": section,
                    "prompt_id": prompt_id,
                    "example_type": example_type,
                    "examples_generated": len(examples),
                    "raw_response_length": len(raw_response),
                }
                with open(log_path, "a", encoding="utf-8") as log_f:
                    log_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    logger.info("Source '%s': wrote %d examples to %s", source_key, total_examples, output_path)
    return total_examples


def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT training data from curated sources using Claude Code"
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "sources.yaml",
        help="Path to sources.yaml",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "prompts",
        help="Directory containing prompt template YAML files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
        help="Output directory for per-source JSONL files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts and estimated token counts without invoking Claude",
    )
    parser.add_argument(
        "--source-key",
        type=str,
        default=None,
        help="Process only this source key (for incremental generation)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Run ID is ISO timestamp at script start
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Load sources
    sources = load_sources(args.sources)
    logger.info("Loaded %d sources from %s", len(sources), args.sources)

    # Load prompt templates
    all_templates = load_prompt_templates(args.prompts)
    logger.info("Loaded %d prompt template files from %s", len(all_templates), args.prompts)

    # Build template lookup: source_key -> list of templates
    template_map: dict[str, list[dict]] = {}
    for tpl_file in all_templates:
        src_key = tpl_file.get("source")
        if src_key:
            template_map.setdefault(src_key, []).extend(tpl_file.get("templates", []))

    # Prepare directories
    raw_content_dir = args.sources.parent / "raw_content"
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.sources.parent / "generation_log.jsonl"

    # Filter sources if --source-key is set
    if args.source_key:
        sources = [s for s in sources if s.get("key") == args.source_key]
        if not sources:
            logger.error("No source found with key '%s'", args.source_key)
            sys.exit(1)

    # Process each source serially
    grand_total = 0
    for source in sources:
        source_key = source["key"]
        templates = template_map.get(source_key, [])
        if not templates:
            logger.info("No prompt templates for source '%s', skipping", source_key)
            continue

        count = generate_for_source(
            source=source,
            templates=templates,
            raw_content_dir=raw_content_dir,
            output_dir=args.output,
            log_path=log_path,
            run_id=run_id,
            dry_run=args.dry_run,
        )
        grand_total += count

    if args.dry_run:
        print(f"\n[DRY RUN] Would process {len(sources)} source(s)")
    else:
        logger.info("Generation complete. Total examples: %d", grand_total)


if __name__ == "__main__":
    main()
