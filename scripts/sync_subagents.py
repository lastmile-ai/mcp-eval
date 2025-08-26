#!/usr/bin/env python
"""Sync subagents from src/mcp_eval/data/subagents to docs/subagents/"""

import sys
import shutil
from pathlib import Path


def sync_subagents():
    """Copy subagent markdown files to docs directory."""
    source_dir = Path(__file__).parent.parent / "src/mcp_eval/data/subagents"
    docs_dir = Path(__file__).parent.parent / "docs/subagents"

    # Create docs/subagents directory if it doesn't exist
    docs_dir.mkdir(exist_ok=True)

    # Find all subagent markdown files (excluding README)
    subagent_files = [
        f for f in source_dir.glob("*.md") if not f.name.startswith("README")
    ]

    print(f"Syncing {len(subagent_files)} subagents from {source_dir} to {docs_dir}")

    for source_file in subagent_files:
        # Convert .md to .mdx for docs
        dest_name = source_file.stem + ".mdx"
        dest_file = docs_dir / dest_name

        # Copy the file
        shutil.copy2(source_file, dest_file)
        print(f"  ✓ {source_file.name} -> {dest_name}")

    print(f"\nSynced {len(subagent_files)} subagents to docs/subagents/")
    return True


def verify_subagents():
    """Verify that subagents in docs match those in src."""
    source_dir = Path(__file__).parent.parent / "src/mcp_eval/data/subagents"
    docs_dir = Path(__file__).parent.parent / "docs/subagents"

    if not docs_dir.exists():
        print(f"Docs directory doesn't exist: {docs_dir}")
        print("Run with --sync to create it")
        return False

    # Find all subagent files
    source_files = {
        f.stem: f for f in source_dir.glob("*.md") if not f.name.startswith("README")
    }
    docs_files = {f.stem: f for f in docs_dir.glob("*.mdx")}

    all_good = True

    # Check for missing files
    missing_in_docs = set(source_files.keys()) - set(docs_files.keys())
    if missing_in_docs:
        print("Missing in docs:")
        for name in missing_in_docs:
            print(f"  - {name}")
        all_good = False

    # Check for extra files in docs
    extra_in_docs = set(docs_files.keys()) - set(source_files.keys())
    if extra_in_docs:
        print("Extra in docs (not in src):")
        for name in extra_in_docs:
            print(f"  - {name}")
        all_good = False

    # Check content matches
    for name in source_files.keys() & docs_files.keys():
        source_content = source_files[name].read_text()
        docs_content = docs_files[name].read_text()

        if source_content != docs_content:
            print(f"Content differs: {name}")
            all_good = False

    if all_good:
        print(f"✓ All {len(source_files)} subagents are in sync")
    else:
        print("\nRun with --sync to synchronize")

    return all_good


if __name__ == "__main__":
    if "--sync" in sys.argv:
        sync_subagents()
    else:
        verify_subagents()
