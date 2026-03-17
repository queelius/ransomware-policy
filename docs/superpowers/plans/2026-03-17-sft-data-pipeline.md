# SFT Domain Knowledge Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a principled, re-runnable pipeline that fetches domain knowledge sources, generates ~2,000 SFT training examples via Claude Code, and validates them with full provenance.

**Architecture:** `source_fetcher.py` caches raw content → `generate_sft_data.py` dispatches Claude Code via `claude -p` to generate examples per prompt template → `merge_and_validate.py` deduplicates, validates, computes coverage stats, and produces the final JSONL dataset with train/eval split.

**Tech Stack:** Python 3.10+, PyYAML, `claude` CLI (for generation), pdfplumber (PDF extraction), requests (web fetch)

**Spec:** `docs/superpowers/specs/2026-03-17-sft-data-pipeline-design.md`

---

## Chunk 1: Infrastructure and Source Fetcher

### Task 1: Directory structure and .gitignore

**Files:**
- Create: `fine-tuning/.gitignore`
- Create: `fine-tuning/prompts/` (directory)
- Create: `fine-tuning/data/raw/` (directory)
- Create: `fine-tuning/data/raw_content/` (directory)

- [ ] **Step 1: Create directories and .gitignore**

```bash
mkdir -p /home/spinoza/github/ransomware-policy/fine-tuning/{prompts,data/raw,data/raw_content}
```

Write `fine-tuning/.gitignore`:
```
data/raw_content/
data/raw/
```

- [ ] **Step 2: Commit**

```bash
cd /home/spinoza/github/ransomware-policy
git add fine-tuning/.gitignore
git commit -m "chore: add .gitignore for SFT data pipeline cache dirs"
```

---

### Task 2: Source fetcher

Fetches and caches raw content from each source in `sources.yaml`. Writes a `manifest.json` with content hashes.

**Files:**
- Create: `fine-tuning/scripts/source_fetcher.py`
- Test: `fine-tuning/scripts/test_source_fetcher.py`

- [ ] **Step 1: Write tests**

```python
# fine-tuning/scripts/test_source_fetcher.py
"""Tests for source_fetcher."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from source_fetcher import (
    load_sources,
    fetch_github_raw,
    compute_content_hash,
    write_manifest,
    SourceConfig,
)


def test_load_sources_returns_training_sources():
    sources = load_sources()
    keys = [s.key for s in sources]
    assert "mitre_attack_stix" in keys
    assert "atomic_red_team" in keys
    # eval-only sources excluded
    assert "malwarebazaar" not in keys


def test_compute_content_hash_deterministic():
    h1 = compute_content_hash(b"hello world")
    h2 = compute_content_hash(b"hello world")
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_write_manifest_creates_valid_json():
    with tempfile.TemporaryDirectory() as d:
        manifest_path = Path(d) / "manifest.json"
        write_manifest(manifest_path, {"mitre_attack_stix": {"hash": "abc", "fetched_at": "2026-03-17"}})
        data = json.loads(manifest_path.read_text())
        assert "mitre_attack_stix" in data
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/spinoza/github/ransomware-policy/fine-tuning/scripts && python -m pytest test_source_fetcher.py -v`

- [ ] **Step 3: Implement source_fetcher.py**

```python
# fine-tuning/scripts/source_fetcher.py
"""Fetch and cache raw content from domain knowledge sources.

Usage:
    python source_fetcher.py [--sources ../data/sources.yaml] [--output ../data/raw_content/]
    python source_fetcher.py --source-key mitre_attack_stix  # fetch single source
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml


@dataclass
class SourceConfig:
    """A single source from sources.yaml."""
    key: str
    name: str
    url: str
    source_type: str  # knowledge_base, real_traces, synthetic
    use: str          # training, evaluation, both
    license: str
    topics: list[str]
    fmt: str
    notes: str


def load_sources(
    path: str | Path = None,
    filter_use: str = "training",
) -> list[SourceConfig]:
    """Load sources.yaml and return training-eligible sources."""
    if path is None:
        path = Path(__file__).parent.parent / "data" / "sources.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    sources = []
    for s in data["sources"]:
        use = s.get("use", "training")
        if filter_use and use not in (filter_use, "both"):
            continue
        sources.append(SourceConfig(
            key=s["key"],
            name=s["name"],
            url=s["url"],
            source_type=s.get("type", "knowledge_base"),
            use=use,
            license=s.get("license", "unknown"),
            topics=s.get("topics", []),
            fmt=s.get("format", "text"),
            notes=s.get("notes", ""),
        ))
    return sources


def compute_content_hash(content: bytes) -> str:
    """SHA-256 hash of content."""
    return hashlib.sha256(content).hexdigest()


def fetch_github_raw(repo: str, paths: list[str], output_dir: Path) -> list[Path]:
    """Fetch specific files from a GitHub repo via raw.githubusercontent.com."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fetched = []
    for p in paths:
        url = f"https://raw.githubusercontent.com/{repo}/main/{p}"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            out = output_dir / Path(p).name
            out.write_bytes(resp.content)
            fetched.append(out)
    return fetched


def fetch_web_page(url: str, output_dir: Path, filename: str) -> Path | None:
    """Fetch a web page and save as text."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "ransomware-research-bot/1.0"})
        if resp.status_code == 200:
            out = output_dir / filename
            out.write_text(resp.text, encoding="utf-8")
            return out
    except requests.RequestException:
        pass
    return None


def fetch_local(source_path: str, output_dir: Path, repo_root: Path) -> list[Path]:
    """Copy local source files to cache."""
    output_dir.mkdir(parents=True, exist_ok=True)
    src = repo_root / source_path
    fetched = []
    if src.is_file():
        out = output_dir / src.name
        out.write_text(src.read_text())
        fetched.append(out)
    elif src.is_dir():
        for f in sorted(src.rglob("*.py")) + sorted(src.rglob("*.md")):
            out = output_dir / f.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(f.read_text())
            fetched.append(out)
    return fetched


def write_manifest(path: Path, entries: dict) -> None:
    """Write manifest.json with fetch metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)


def fetch_source(source: SourceConfig, output_base: Path, repo_root: Path) -> dict:
    """Fetch a single source and return manifest entry."""
    output_dir = output_base / source.key
    now = datetime.now(timezone.utc).isoformat()
    entry = {"key": source.key, "name": source.name, "url": source.url, "fetched_at": now, "files": []}

    if "github.com" in source.url:
        # GitHub source — fetch key files
        if "attack-stix-data" in source.url:
            files = fetch_github_raw(
                "mitre-attack/attack-stix-data",
                ["enterprise-attack/enterprise-attack.json"],
                output_dir,
            )
        elif "atomic-red-team" in source.url:
            # Fetch the index and key technique dirs
            files = fetch_github_raw(
                "redcanaryco/atomic-red-team",
                ["atomics/Indexes/Indexes-Markdown/index.md"],
                output_dir,
            )
        else:
            files = []
        entry["files"] = [str(f.relative_to(output_base)) for f in files]

    elif source.url.startswith(("http://", "https://")):
        # Web source
        result = fetch_web_page(source.url, output_dir, f"{source.key}.html")
        if result:
            entry["files"] = [str(result.relative_to(output_base))]

    elif not source.url.startswith("http"):
        # Local source
        files = fetch_local(source.url, output_dir, repo_root)
        entry["files"] = [str(f.relative_to(output_base)) for f in files]

    # Compute hashes
    for f_rel in entry["files"]:
        f_path = output_base / f_rel
        if f_path.exists():
            entry.setdefault("hashes", {})[f_rel] = compute_content_hash(f_path.read_bytes())

    return entry


def main():
    parser = argparse.ArgumentParser(description="Fetch and cache source content")
    parser.add_argument("--sources", default=None, help="Path to sources.yaml")
    parser.add_argument("--output", default=None, help="Output directory for cached content")
    parser.add_argument("--source-key", default=None, help="Fetch only this source key")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent  # ransomware-policy/
    output_base = Path(args.output) if args.output else Path(__file__).parent.parent / "data" / "raw_content"
    sources = load_sources(args.sources)

    if args.source_key:
        sources = [s for s in sources if s.key == args.source_key]
        if not sources:
            print(f"Source key '{args.source_key}' not found in sources.yaml")
            sys.exit(1)

    manifest_entries = {}
    for source in sources:
        print(f"Fetching {source.key} ({source.name})...")
        entry = fetch_source(source, output_base, repo_root)
        manifest_entries[source.key] = entry
        print(f"  → {len(entry.get('files', []))} files cached")

    manifest_path = output_base / "manifest.json"
    write_manifest(manifest_path, manifest_entries)
    print(f"\nManifest written to {manifest_path}")
    print(f"Total: {len(manifest_entries)} sources fetched")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `cd /home/spinoza/github/ransomware-policy/fine-tuning/scripts && pip install pyyaml requests && python -m pytest test_source_fetcher.py -v`

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/ransomware-policy
git add fine-tuning/scripts/source_fetcher.py fine-tuning/scripts/test_source_fetcher.py
git commit -m "feat: add source fetcher for SFT data pipeline"
```

---

### Task 3: Prompt templates (8 YAML files)

One per training source. These define how Claude Code generates examples from each source's content.

**Files:**
- Create: `fine-tuning/prompts/mitre_attack.yaml`
- Create: `fine-tuning/prompts/atomic_red_team.yaml`
- Create: `fine-tuning/prompts/cisa_guide.yaml`
- Create: `fine-tuning/prompts/dfir_report.yaml`
- Create: `fine-tuning/prompts/windows_event_log.yaml`
- Create: `fine-tuning/prompts/kaspersky_securelist.yaml`
- Create: `fine-tuning/prompts/sans_dfir.yaml`
- Create: `fine-tuning/prompts/prompting_only.yaml`

- [ ] **Step 1: Write all 8 prompt templates**

Each YAML follows the schema from the spec (Section 2.3): `source`, `description`, `templates` array with `id`, `type`, `system`, `prompt`, `target_count`.

Key design principles for each template:
- **system prompt** is always: `"You are a ransomware forensic analyst with deep expertise in Windows internals and incident response."`
- **QA prompts** ask for 3-5 question/answer pairs per content chunk
- **Instruction prompts** ask for analyst-style "analyze/detect/investigate" scenarios
- **Passage prompts** ask for domain-rich explanatory text
- All prompts instruct output as JSON array of `{"question": "...", "answer": "..."}` for QA, or `{"instruction": "...", "response": "..."}` for instruction/passage
- Each prompt references `{content}` placeholder that gets filled with the source chunk

Example — `mitre_attack.yaml`:
```yaml
source: mitre_attack_stix
description: "Generate QA and instruction examples from ATT&CK technique descriptions"

templates:
  - id: technique_qa_v1
    type: qa
    system: "You are a ransomware forensic analyst with deep expertise in Windows internals and incident response."
    prompt: |
      Given this ATT&CK technique, generate 3-5 QA pairs that would help a small language
      model understand ransomware detection. Cover: what the technique is, how to detect it,
      what artifacts it leaves on a Windows host, and how ransomware groups use it.

      {content}

      Output as a JSON array: [{"question": "...", "answer": "..."}]
      Each answer should be 2-4 sentences, technically precise.
    target_count: 3-5

  - id: technique_instruction_v1
    type: instruction
    system: "You are a ransomware forensic analyst with deep expertise in Windows internals and incident response."
    prompt: |
      Based on this ATT&CK technique, write an instruction-following training example where
      a security analyst asks how to detect or investigate this technique on a Windows endpoint.
      The response should cover specific registry keys, event log IDs, file system artifacts,
      and process behaviors to look for.

      {content}

      Output as JSON: [{"instruction": "...", "response": "..."}]
      The response should be 4-8 sentences with specific technical details.
    target_count: 1
```

Create similar templates for each source, adjusting the prompt to match the content format:
- `atomic_red_team.yaml`: Generate from YAML test definitions — focus on commands, execution, and detection
- `cisa_guide.yaml`: QA + instruction + passage from incident response procedures
- `dfir_report.yaml`: Instruction + passage from case study narratives
- `windows_event_log.yaml`: QA about specific event IDs and their forensic significance
- `kaspersky_securelist.yaml`: Instruction + passage from group TTP analysis
- `sans_dfir.yaml`: QA from cheat sheet entries (artifact locations, forensic patterns)
- `prompting_only.yaml`: Instruction from expert annotations and multi-layer predictions

- [ ] **Step 2: Commit**

```bash
cd /home/spinoza/github/ransomware-policy
git add fine-tuning/prompts/*.yaml
git commit -m "feat: add prompt templates for SFT data generation"
```

---

## Chunk 2: Generation and Validation

### Task 4: Generation orchestrator

Reads sources and prompts, dispatches Claude Code via `claude -p`, writes per-source JSONL with metadata.

**Files:**
- Create: `fine-tuning/scripts/generate_sft_data.py`
- Test: `fine-tuning/scripts/test_generate_sft_data.py`

- [ ] **Step 1: Write tests**

```python
# fine-tuning/scripts/test_generate_sft_data.py
"""Tests for generate_sft_data."""
import json
import tempfile
from pathlib import Path
from generate_sft_data import (
    load_prompt_templates,
    format_prompt,
    parse_generated_examples,
    build_metadata,
    SFTExample,
)


def test_load_prompt_templates():
    templates = load_prompt_templates()
    assert len(templates) > 0
    assert any(t["source"] == "mitre_attack_stix" for t in templates)


def test_format_prompt_substitutes_content():
    template = {"prompt": "Analyze this:\n{content}\nOutput JSON.", "system": "You are an analyst."}
    result = format_prompt(template, content="Test content here")
    assert "Test content here" in result
    assert "{content}" not in result


def test_parse_generated_examples_valid_json():
    raw = '[{"question": "What is T1486?", "answer": "Data Encrypted for Impact."}]'
    examples = parse_generated_examples(raw, example_type="qa")
    assert len(examples) == 1
    assert examples[0].messages[1]["content"] == "What is T1486?"


def test_parse_generated_examples_handles_malformed():
    examples = parse_generated_examples("not json at all", example_type="qa")
    assert examples == []


def test_build_metadata():
    meta = build_metadata(
        source_key="mitre_attack_stix",
        source_url="https://github.com/mitre-attack/attack-stix-data",
        source_section="T1486",
        example_type="qa",
        topics=["ransomware_ttps"],
        prompt_id="technique_qa_v1",
        run_id="2026-03-17T00:00:00Z",
    )
    assert meta["source"] == "mitre_attack_stix"
    assert meta["generation_prompt_id"] == "technique_qa_v1"
```

- [ ] **Step 2: Implement generate_sft_data.py**

The orchestrator:
1. Loads `sources.yaml` (training sources only) and `prompts/*.yaml`
2. For each source, reads cached content from `data/raw_content/{key}/`
3. Chunks content (one technique, one article, one section at a time)
4. For each chunk × template: formats the prompt, invokes `claude -p`, parses response
5. Writes examples with metadata to `data/raw/{key}.jsonl`
6. Writes generation log to `data/generation_log.jsonl`
7. `--dry-run` mode: prints prompts and token estimates without invoking

Key functions:
- `load_prompt_templates(prompts_dir) -> list[dict]`
- `format_prompt(template, content) -> str`
- `invoke_claude(prompt, system) -> str` — calls `subprocess.run(["claude", "-p", prompt, "--output-format", "text"])`
- `parse_generated_examples(raw_response, example_type) -> list[SFTExample]`
- `build_metadata(source_key, source_url, ...) -> dict`
- `SFTExample` dataclass: `messages: list[dict]`, `metadata: dict`

The `invoke_claude` function is the core — it shells out to `claude -p` with the formatted prompt. Output is parsed as JSON. Retry once on failure.

- [ ] **Step 3: Run tests**

Run: `cd /home/spinoza/github/ransomware-policy/fine-tuning/scripts && python -m pytest test_generate_sft_data.py -v`

- [ ] **Step 4: Commit**

```bash
cd /home/spinoza/github/ransomware-policy
git add fine-tuning/scripts/generate_sft_data.py fine-tuning/scripts/test_generate_sft_data.py
git commit -m "feat: add SFT data generation orchestrator"
```

---

### Task 5: Merge and validate

Reads per-source JSONL files, deduplicates, validates, computes stats, produces final dataset with train/eval split.

**Files:**
- Create: `fine-tuning/scripts/merge_and_validate.py`
- Test: `fine-tuning/scripts/test_merge_and_validate.py`

- [ ] **Step 1: Write tests**

```python
# fine-tuning/scripts/test_merge_and_validate.py
"""Tests for merge_and_validate."""
import json
import tempfile
from pathlib import Path
from merge_and_validate import (
    load_raw_examples,
    validate_example,
    exact_dedup,
    jaccard_similarity,
    compute_coverage_stats,
    split_train_eval,
)


VALID_EXAMPLE = {
    "messages": [
        {"role": "system", "content": "You are an analyst."},
        {"role": "user", "content": "What is T1486?"},
        {"role": "assistant", "content": "T1486 is Data Encrypted for Impact, a technique used by ransomware."},
    ],
    "metadata": {
        "source": "mitre_attack_stix",
        "source_url": "https://github.com/mitre-attack/attack-stix-data",
        "source_section": "T1486",
        "example_type": "qa",
        "topics": ["ransomware_ttps"],
        "generation_prompt_id": "technique_qa_v1",
        "generated_at": "2026-03-17T14:30:00Z",
        "generator_model": "claude-opus-4-6",
        "quality_score": None,
    },
}


def test_validate_example_accepts_valid():
    errors = validate_example(VALID_EXAMPLE)
    assert errors == []


def test_validate_example_rejects_empty_assistant():
    bad = json.loads(json.dumps(VALID_EXAMPLE))
    bad["messages"][2]["content"] = ""
    errors = validate_example(bad)
    assert len(errors) > 0


def test_validate_example_rejects_short_assistant():
    bad = json.loads(json.dumps(VALID_EXAMPLE))
    bad["messages"][2]["content"] = "Yes."
    errors = validate_example(bad)
    assert any("50 characters" in e for e in errors)


def test_exact_dedup_removes_duplicates():
    examples = [VALID_EXAMPLE, VALID_EXAMPLE, VALID_EXAMPLE]
    deduped, removed = exact_dedup(examples)
    assert len(deduped) == 1
    assert removed == 2


def test_jaccard_similarity():
    assert jaccard_similarity("hello world foo", "hello world bar") > 0.5
    assert jaccard_similarity("completely different", "nothing alike here") < 0.5


def test_split_train_eval_proportions():
    examples = [VALID_EXAMPLE] * 100
    train, eval_ = split_train_eval(examples, eval_fraction=0.1, seed=42)
    assert len(train) == 90
    assert len(eval_) == 10
```

- [ ] **Step 2: Implement merge_and_validate.py**

Key functions:
- `load_raw_examples(raw_dir) -> list[dict]` — reads all `data/raw/*.jsonl`
- `validate_example(example) -> list[str]` — returns list of error strings (empty = valid)
- `exact_dedup(examples) -> tuple[list, int]` — hash-based exact dedup on user+assistant content
- `jaccard_similarity(text_a, text_b) -> float` — word-level Jaccard for fuzzy dedup
- `fuzzy_dedup(examples, threshold=0.85) -> tuple[list, list]` — flags near-duplicates
- `compute_coverage_stats(examples) -> dict` — counts by source, type, topic
- `split_train_eval(examples, eval_fraction=0.1, seed=42) -> tuple[list, list]` — stratified split
- `format_coverage_report(stats) -> str` — markdown report

CLI: `python merge_and_validate.py [--raw-dir ../data/raw] [--output-dir ../data]`

Produces:
- `data/sft_domain_knowledge_train.jsonl`
- `data/sft_domain_knowledge_eval.jsonl`
- `data/coverage_report.md`
- `data/validation_log.jsonl`

- [ ] **Step 3: Run tests**

Run: `cd /home/spinoza/github/ransomware-policy/fine-tuning/scripts && python -m pytest test_merge_and_validate.py -v`

- [ ] **Step 4: Commit**

```bash
cd /home/spinoza/github/ransomware-policy
git add fine-tuning/scripts/merge_and_validate.py fine-tuning/scripts/test_merge_and_validate.py
git commit -m "feat: add merge and validation for SFT data pipeline"
```

---

## Chunk 3: Training Script

### Task 6: SFT training script for chat-format JSONL

New training script that loads the chat-format dataset and fine-tunes Qwen3-4B/1.7B with QLoRA.

**Files:**
- Create: `fine-tuning/scripts/train_sft.py`

- [ ] **Step 1: Write train_sft.py**

A focused script using TRL's `SFTTrainer` with QLoRA:

```python
# fine-tuning/scripts/train_sft.py
"""Domain knowledge SFT training script.

Usage:
    python train_sft.py --model Qwen/Qwen3-4B --data ../data/sft_domain_knowledge_train.jsonl
    python train_sft.py --model Qwen/Qwen3-1.7B --data ../data/sft_domain_knowledge_train.jsonl --epochs 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import torch


def load_chat_dataset(path: str) -> Dataset:
    """Load chat-format JSONL as a HuggingFace Dataset."""
    examples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            examples.append({"messages": data["messages"]})
    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser(description="Domain knowledge SFT")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--eval-data", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./checkpoints-sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    # Load dataset
    train_dataset = load_chat_dataset(args.data)
    eval_dataset = load_chat_dataset(args.eval_data) if args.eval_data else None
    print(f"Train: {len(train_dataset)} examples")
    if eval_dataset:
        print(f"Eval: {len(eval_dataset)} examples")

    # QLoRA 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model + tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training config
    training_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        max_seq_length=args.max_seq_length,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
cd /home/spinoza/github/ransomware-policy
git add fine-tuning/scripts/train_sft.py
git commit -m "feat: add SFT training script for chat-format domain knowledge"
```

---

## Chunk 4: Run the Pipeline

### Task 7: Fetch sources and generate data

This is the actual data generation run. Uses the pipeline scripts built in Tasks 1-5.

- [ ] **Step 1: Fetch source content**

```bash
cd /home/spinoza/github/ransomware-policy/fine-tuning
python scripts/source_fetcher.py
```

Verify manifest: `cat data/raw_content/manifest.json | python -m json.tool | head -30`

- [ ] **Step 2: Dry run the generator**

```bash
python scripts/generate_sft_data.py --dry-run
```

Review the prompts that would be sent, verify token estimates are reasonable.

- [ ] **Step 3: Run the full generation**

```bash
python scripts/generate_sft_data.py
```

This dispatches `claude -p` for each content chunk × prompt template. Serial, one at a time. Monitor progress. Expected: ~100-200 Claude invocations producing ~2,000 examples across 8 sources.

- [ ] **Step 4: Merge and validate**

```bash
python scripts/merge_and_validate.py
```

Review output:
- `data/coverage_report.md` — check source balance, type distribution
- `data/validation_log.jsonl` — check what was dropped and why
- `data/sft_domain_knowledge_train.jsonl` — spot-check 10 examples
- `data/sft_domain_knowledge_eval.jsonl` — verify eval set

- [ ] **Step 5: Commit the dataset and provenance**

```bash
cd /home/spinoza/github/ransomware-policy
git add fine-tuning/data/sft_domain_knowledge_train.jsonl \
        fine-tuning/data/sft_domain_knowledge_eval.jsonl \
        fine-tuning/data/generation_log.jsonl \
        fine-tuning/data/coverage_report.md
git commit -m "feat: generate domain knowledge SFT dataset (~2K examples)"
```

---

## Final Verification

- [ ] **Verify provenance chain**: Pick 3 random examples from the final dataset. For each, trace back through `metadata.source` → `sources.yaml`, `metadata.generation_prompt_id` → `prompts/*.yaml`, `metadata.generated_at` → `generation_log.jsonl`, raw content → `raw_content/manifest.json`.

- [ ] **Verify train/eval split**: Check that both files exist, sizes are ~90/10, and sources are represented in both.

- [ ] **Run all fine-tuning tests**:

```bash
cd /home/spinoza/github/ransomware-policy/fine-tuning/scripts
python -m pytest test_source_fetcher.py test_generate_sft_data.py test_merge_and_validate.py -v
```
