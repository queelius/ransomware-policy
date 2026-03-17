#!/usr/bin/env python3
"""Fetch and cache raw content from domain knowledge sources defined in sources.yaml.

Fetches sources marked use=training or use=both, caches to
fine-tuning/data/raw_content/{source_key}/, and writes a manifest.json
with fetch timestamps and SHA-256 content hashes.

Usage:
    python source_fetcher.py [--sources PATH] [--output PATH] [--source-key KEY]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class SourceConfig:
    """One entry from sources.yaml."""

    key: str
    name: str
    url: str
    source_type: str
    use: str
    license: str
    topics: list[str] = field(default_factory=list)
    fmt: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

def load_sources(path: str | Path, filter_use: str = "training") -> list[SourceConfig]:
    """Load sources.yaml and return entries whose *use* matches *filter_use* or 'both'.

    Parameters
    ----------
    path : path to sources.yaml
    filter_use : keep entries where ``use == filter_use`` or ``use == 'both'``
    """
    path = Path(path)
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)

    sources: list[SourceConfig] = []
    for entry in data.get("sources", []):
        use_val = entry.get("use", "")
        if use_val not in (filter_use, "both"):
            continue
        sources.append(
            SourceConfig(
                key=entry["key"],
                name=entry["name"],
                url=entry.get("url", ""),
                source_type=entry.get("type", ""),
                use=use_val,
                license=entry.get("license", ""),
                topics=entry.get("topics", []),
                fmt=entry.get("format", ""),
                notes=entry.get("notes", ""),
            )
        )
    return sources


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def compute_content_hash(content: bytes) -> str:
    """Return the SHA-256 hex digest of *content*."""
    return hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# Fetchers per source type
# ---------------------------------------------------------------------------

_GITHUB_RAW = "https://raw.githubusercontent.com"
_REQUEST_TIMEOUT = 30  # seconds

# Atomic Red Team technique files relevant to ransomware investigation.
_ATOMIC_TECHNIQUES = ["T1486", "T1490", "T1059", "T1547", "T1562"]


def _fetch_github_mitre(source: SourceConfig, dest: Path) -> list[dict[str, Any]]:
    """Download MITRE ATT&CK STIX enterprise-attack.json."""
    url = f"{_GITHUB_RAW}/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
    logger.info("Fetching %s", url)
    resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    out_path = dest / "enterprise-attack.json"
    out_path.write_bytes(resp.content)
    return [
        {
            "file": str(out_path.name),
            "url": url,
            "sha256": compute_content_hash(resp.content),
            "bytes": len(resp.content),
        }
    ]


def _fetch_github_atomic(source: SourceConfig, dest: Path) -> list[dict[str, Any]]:
    """Download Atomic Red Team index and key technique YAMLs."""
    files: list[dict[str, Any]] = []

    # Index markdown
    index_url = f"{_GITHUB_RAW}/redcanaryco/atomic-red-team/master/atomics/Indexes/Indexes-Markdown/index.md"
    logger.info("Fetching %s", index_url)
    resp = requests.get(index_url, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    out = dest / "index.md"
    out.write_bytes(resp.content)
    files.append(
        {
            "file": str(out.name),
            "url": index_url,
            "sha256": compute_content_hash(resp.content),
            "bytes": len(resp.content),
        }
    )

    # Key technique YAMLs
    for tid in _ATOMIC_TECHNIQUES:
        yaml_url = f"{_GITHUB_RAW}/redcanaryco/atomic-red-team/master/atomics/{tid}/{tid}.yaml"
        logger.info("Fetching %s", yaml_url)
        try:
            resp = requests.get(yaml_url, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.HTTPError:
            logger.warning("Failed to fetch %s (HTTP %s), skipping", yaml_url, resp.status_code)
            continue
        fname = f"{tid}.yaml"
        (dest / fname).write_bytes(resp.content)
        files.append(
            {
                "file": fname,
                "url": yaml_url,
                "sha256": compute_content_hash(resp.content),
                "bytes": len(resp.content),
            }
        )

    return files


def _fetch_webpage(source: SourceConfig, dest: Path) -> list[dict[str, Any]]:
    """Fetch a web page (HTML) via requests."""
    logger.info("Fetching %s", source.url)
    resp = requests.get(
        source.url,
        timeout=_REQUEST_TIMEOUT,
        headers={"User-Agent": "ransomware-policy-research/0.1 (academic)"},
    )
    resp.raise_for_status()
    fname = f"{source.key}.html"
    out = dest / fname
    out.write_bytes(resp.content)
    return [
        {
            "file": fname,
            "url": source.url,
            "sha256": compute_content_hash(resp.content),
            "bytes": len(resp.content),
        }
    ]


def _fetch_pdf(source: SourceConfig, dest: Path) -> list[dict[str, Any]]:
    """Handle PDF sources (SANS DFIR requires manual download)."""
    cached = dest / f"{source.key}.pdf"
    if cached.exists():
        content = cached.read_bytes()
        logger.info("Using cached PDF %s", cached)
        return [
            {
                "file": cached.name,
                "url": source.url,
                "sha256": compute_content_hash(content),
                "bytes": len(content),
                "cached": True,
            }
        ]
    logger.warning(
        "PDF source '%s' requires manual download (registration needed). "
        "Place the file at %s and re-run.",
        source.name,
        cached,
    )
    return [
        {
            "file": cached.name,
            "url": source.url,
            "sha256": None,
            "bytes": 0,
            "skipped": True,
            "reason": "manual download required",
        }
    ]


def _fetch_local(source: SourceConfig, dest: Path, repo_root: Path) -> list[dict[str, Any]]:
    """Copy content from a local repo path."""
    src = repo_root / source.url
    if not src.exists():
        logger.warning("Local source path does not exist: %s", src)
        return [
            {
                "file": source.url,
                "url": source.url,
                "sha256": None,
                "bytes": 0,
                "skipped": True,
                "reason": f"path not found: {src}",
            }
        ]

    files: list[dict[str, Any]] = []
    if src.is_file():
        content = src.read_bytes()
        fname = src.name
        (dest / fname).write_bytes(content)
        files.append(
            {
                "file": fname,
                "url": source.url,
                "sha256": compute_content_hash(content),
                "bytes": len(content),
            }
        )
    elif src.is_dir():
        for child in sorted(src.rglob("*")):
            if child.is_file():
                rel = child.relative_to(src)
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                content = child.read_bytes()
                shutil.copy2(child, out)
                files.append(
                    {
                        "file": str(rel),
                        "url": source.url,
                        "sha256": compute_content_hash(content),
                        "bytes": len(content),
                    }
                )
    return files


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# Map source keys to specific GitHub fetchers
_GITHUB_FETCHERS = {
    "mitre_attack_stix": _fetch_github_mitre,
    "atomic_red_team": _fetch_github_atomic,
}


def fetch_source(
    source: SourceConfig,
    output_base: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Fetch a single source and return a manifest entry dict.

    Parameters
    ----------
    source : the SourceConfig to fetch
    output_base : base directory for cached content (e.g. fine-tuning/data/raw_content)
    repo_root : root of the repository (for resolving local paths)

    Returns
    -------
    dict with keys: key, name, fetched_at, files (list of per-file info)
    """
    output_base = Path(output_base)
    repo_root = Path(repo_root)
    dest = output_base / source.key
    dest.mkdir(parents=True, exist_ok=True)

    fetched_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    try:
        if source.key in _GITHUB_FETCHERS:
            file_entries = _GITHUB_FETCHERS[source.key](source, dest)
        elif source.source_type == "knowledge_base" and source.fmt == "text":
            file_entries = _fetch_webpage(source, dest)
        elif source.fmt == "text" and "pdf" not in source.notes.lower():
            file_entries = _fetch_webpage(source, dest)
        elif "pdf" in source.fmt.lower() or source.key == "sans_dfir":
            file_entries = _fetch_pdf(source, dest)
        elif source.source_type == "synthetic" or source.url.startswith(("prompting-only/", "active-detective/")):
            file_entries = _fetch_local(source, dest, repo_root)
        else:
            # Fallback: try as webpage
            file_entries = _fetch_webpage(source, dest)
    except requests.RequestException as exc:
        logger.error("Network error fetching '%s': %s", source.key, exc)
        file_entries = [
            {
                "file": None,
                "url": source.url,
                "sha256": None,
                "bytes": 0,
                "skipped": True,
                "reason": str(exc),
            }
        ]

    return {
        "key": source.key,
        "name": source.name,
        "fetched_at": fetched_at,
        "files": file_entries,
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(path: str | Path, entries: list[dict[str, Any]]) -> None:
    """Write *entries* as a JSON manifest file at *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(
            {
                "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "sources": entries,
            },
            fh,
            indent=2,
        )
    logger.info("Manifest written to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_repo_root() -> Path:
    """Walk up from this script to find the repo root (contains .git)."""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    # Fallback: two dirs up from this script
    return Path(__file__).resolve().parent.parent.parent


def main(argv: list[str] | None = None) -> None:
    repo_root = _resolve_repo_root()
    default_sources = repo_root / "fine-tuning" / "data" / "sources.yaml"
    default_output = repo_root / "fine-tuning" / "data" / "raw_content"

    parser = argparse.ArgumentParser(description="Fetch and cache SFT data sources")
    parser.add_argument(
        "--sources",
        type=Path,
        default=default_sources,
        help="Path to sources.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output directory for cached content (default: %(default)s)",
    )
    parser.add_argument(
        "--source-key",
        type=str,
        default=None,
        help="Fetch only a single source by key",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sources = load_sources(args.sources, filter_use="training")
    if args.source_key:
        sources = [s for s in sources if s.key == args.source_key]
        if not sources:
            logger.error("No source found with key '%s' (with use=training or use=both)", args.source_key)
            return

    logger.info("Fetching %d source(s)...", len(sources))

    entries: list[dict[str, Any]] = []
    for source in sources:
        logger.info("--- %s ---", source.key)
        entry = fetch_source(source, args.output, repo_root)
        entries.append(entry)
        total_bytes = sum(f.get("bytes", 0) for f in entry["files"])
        skipped = any(f.get("skipped") for f in entry["files"])
        if skipped:
            logger.info("  => skipped (see reason in manifest)")
        else:
            logger.info("  => %d file(s), %d bytes total", len(entry["files"]), total_bytes)

    manifest_path = args.output / "manifest.json"
    write_manifest(manifest_path, entries)
    logger.info("Done. Manifest at %s", manifest_path)


if __name__ == "__main__":
    main()
