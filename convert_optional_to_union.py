#!/usr/bin/env python3
"""
Script to convert Optional[type] to type | None in Python files.
Handles nested Optional types and preserves formatting.
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple


def convert_optional_to_union(content: str) -> Tuple[str, int]:
    """
    Convert all instances of Optional[type] to type | None.
    
    Args:
        content: The file content as a string
        
    Returns:
        Tuple of (modified content, number of replacements made)
    """
    count = 0
    
    # Pattern to match Optional[...] with potentially nested brackets
    # This uses a recursive approach to handle nested brackets
    def replace_optional(match):
        nonlocal count
        count += 1
        inner = match.group(1)
        return f"{inner} | None"
    
    # First, handle simple cases without nested brackets
    pattern_simple = r'\bOptional\[([^\[\]]+)\]'
    
    # Keep replacing until no more matches (handles nested cases)
    prev_content = content
    while True:
        # Replace innermost Optional first
        content = re.sub(pattern_simple, replace_optional, content)
        
        # Also handle Optional with nested brackets using a more complex approach
        # This pattern matches Optional[...] where ... can contain balanced brackets
        pattern_complex = r'\bOptional\[((?:[^\[\]]|\[[^\[\]]*\])+)\]'
        content = re.sub(pattern_complex, replace_optional, content)
        
        if content == prev_content:
            break
        prev_content = content
    
    return content, count


def process_file(file_path: Path, dry_run: bool = False) -> int:
    """
    Process a single Python file.
    
    Args:
        file_path: Path to the Python file
        dry_run: If True, don't write changes, just report them
        
    Returns:
        Number of replacements made
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return 0
    
    # Check if file imports Optional from typing
    if 'Optional' not in original_content:
        return 0
    
    modified_content, count = convert_optional_to_union(original_content)
    
    if count > 0:
        print(f"{'[DRY RUN] ' if dry_run else ''}Modified {file_path}: {count} replacement(s)")
        
        if not dry_run:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
            except Exception as e:
                print(f"Error writing {file_path}: {e}", file=sys.stderr)
                return 0
    
    return count


def find_python_files(paths: List[str], recursive: bool = True) -> List[Path]:
    """
    Find all Python files in the given paths.
    
    Args:
        paths: List of file or directory paths
        recursive: If True, search directories recursively
        
    Returns:
        List of Python file paths
    """
    python_files = []
    
    for path_str in paths:
        path = Path(path_str)
        
        if path.is_file():
            if path.suffix == '.py':
                python_files.append(path)
        elif path.is_dir():
            if recursive:
                python_files.extend(path.rglob('*.py'))
            else:
                python_files.extend(path.glob('*.py'))
        else:
            print(f"Warning: {path} not found", file=sys.stderr)
    
    return python_files


def main():
    parser = argparse.ArgumentParser(
        description='Convert Optional[type] to type | None in Python files'
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='Files or directories to process'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search directories recursively'
    )
    
    args = parser.parse_args()
    
    # Find all Python files
    python_files = find_python_files(args.paths, recursive=not args.no_recursive)
    
    if not python_files:
        print("No Python files found to process")
        return 1
    
    print(f"Processing {len(python_files)} Python file(s)...")
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
    
    total_replacements = 0
    modified_files = 0
    
    for file_path in python_files:
        count = process_file(file_path, dry_run=args.dry_run)
        if count > 0:
            modified_files += 1
            total_replacements += count
    
    print(f"\nSummary:")
    print(f"  Files scanned: {len(python_files)}")
    print(f"  Files modified: {modified_files}")
    print(f"  Total replacements: {total_replacements}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())