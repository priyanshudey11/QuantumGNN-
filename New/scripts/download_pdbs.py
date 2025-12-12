"""Download PDB files (helper utility)."""

import os
import urllib.request
from pathlib import Path
import argparse
from typing import List


def download_pdb(pdb_id: str, output_dir: str = "data/pdb"):
    """Download a single PDB file from RCSB PDB."""
    
    pdb_id = pdb_id.lower().strip()
    output_path = Path(output_dir) / f"{pdb_id}.pdb"
    
    if output_path.exists():
        print(f"✓ {pdb_id}.pdb already exists")
        return True
    
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    try:
        print(f"Downloading {pdb_id}...", end=" ")
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Saved to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def download_pdb_list(pdb_ids: List[str], output_dir: str = "data/pdb"):
    """Download multiple PDB files."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for pdb_id in pdb_ids:
        if download_pdb(pdb_id, output_dir):
            successful += 1
        else:
            failed += 1
    
    print(f"\nSummary: {successful} successful, {failed} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PDB files")
    parser.add_argument("--pdb-ids", type=str, nargs="+", help="PDB IDs to download")
    parser.add_argument("--file", type=str, help="File with PDB IDs (one per line)")
    parser.add_argument("--output-dir", type=str, default="data/pdb", help="Output directory")
    
    args = parser.parse_args()
    
    pdb_ids = []
    
    if args.pdb_ids:
        pdb_ids.extend(args.pdb_ids)
    
    if args.file:
        with open(args.file, 'r') as f:
            pdb_ids.extend([line.strip() for line in f if line.strip()])
    
    if not pdb_ids:
        print("Specify PDB IDs with --pdb-ids or --file")
        exit(1)
    
    download_pdb_list(pdb_ids, args.output_dir)
