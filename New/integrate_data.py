"""Integration script for alternate data source."""

import argparse
import sys
from pathlib import Path
from src.data.alternate_loader import AlternateDataLoader
from src.data.data_adapter import AlternateDataAdapter


def main():
    parser = argparse.ArgumentParser(
        description="Integrate alternate data source into pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load and analyze alternate data
  python integrate_data.py /path/to/data --summary
  
  # Create drug and protein lookups
  python integrate_data.py /path/to/data --export
  
  # Validate data integrity
  python integrate_data.py /path/to/data --validate
  
  # Query specific complexes
  python integrate_data.py /path/to/data --pdb 4hvb
  python integrate_data.py /path/to/data --uniprot P48736
        """
    )

    parser.add_argument('data_path', type=str,
                       help='Path to alternate data directory')
    parser.add_argument('--summary', action='store_true',
                       help='Print data summary')
    parser.add_argument('--export', action='store_true',
                       help='Export complexes, drugs, and proteins to CSV')
    parser.add_argument('--validate', action='store_true',
                       help='Validate data integrity')
    parser.add_argument('--pdb', type=str,
                       help='Query complexes by PDB ID')
    parser.add_argument('--uniprot', type=str,
                       help='Query complexes by UniProt ID')
    parser.add_argument('--ligand', type=str,
                       help='Query complexes by ligand code')
    parser.add_argument('--output-dir', type=str, default='outputs/alternate_data',
                       help='Output directory for exports')
    parser.add_argument('--liganded-only', action='store_true',
                       help='Only include liganded complexes')

    args = parser.parse_args()

    # Validate data path
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data path not found: {args.data_path}")
        return 1

    print(f"Loading alternate data from: {args.data_path}\n")

    # Load data
    try:
        loader = AlternateDataLoader(str(data_path))
        adapter = AlternateDataAdapter(loader)
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    # Print summary
    if args.summary or not any([args.export, args.validate, args.pdb, args.uniprot, args.ligand]):
        adapter.print_summary()

    # Validate
    if args.validate:
        print("\n=== Validation Report ===")
        loader.report_validation()

    # Export
    if args.export:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting to {output_dir}...")
        adapter.save_pairs_csv(str(output_dir / "complexes.csv"))
        adapter.save_drug_lookup_csv(str(output_dir / "drug_lookup.csv"))
        adapter.save_protein_lookup_csv(str(output_dir / "protein_lookup.csv"))
        print("✓ Export complete")

    # Query by PDB
    if args.pdb:
        complexes = loader.get_by_pdb_id(args.pdb)
        if not complexes:
            print(f"No complexes found for PDB ID: {args.pdb}")
        else:
            print(f"\nFound {len(complexes)} complexes for PDB {args.pdb}:")
            for c in complexes[:10]:
                print(f"  {c.pdb_id} chain {c.chain} {c.uniprot_id} "
                      f"ligand={c.ligand_code} interface={bool(c.interface_residues)}")
            if len(complexes) > 10:
                print(f"  ... and {len(complexes) - 10} more")

    # Query by UniProt
    if args.uniprot:
        complexes = loader.get_by_uniprot_id(args.uniprot)
        if not complexes:
            print(f"No complexes found for UniProt ID: {args.uniprot}")
        else:
            print(f"\nFound {len(complexes)} complexes for UniProt {args.uniprot}:")
            for c in complexes[:10]:
                print(f"  {c.pdb_id} chain {c.chain} ligand={c.ligand_code} "
                      f"interface={bool(c.interface_residues)}")
            if len(complexes) > 10:
                print(f"  ... and {len(complexes) - 10} more")

    # Query by ligand
    if args.ligand:
        matching = [c for c in loader.complexes if c.ligand_code == args.ligand]
        if not matching:
            print(f"No complexes found for ligand: {args.ligand}")
        else:
            print(f"\nFound {len(matching)} complexes for ligand {args.ligand}:")
            for c in matching[:10]:
                print(f"  {c.pdb_id} chain {c.chain} {c.uniprot_id}")
            if len(matching) > 10:
                print(f"  ... and {len(matching) - 10} more")

    return 0


if __name__ == '__main__':
    sys.exit(main())
