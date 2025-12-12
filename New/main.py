"""Main CLI entry point."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Quantum Drug-Protein Interaction Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate sample data
  python main.py data --generate-samples
  
  # Train full model
  python main.py train --config configs/default.yaml
  
  # Train classical baseline
  python main.py train --config configs/default.yaml --model-type classical
  
  # Evaluate trained model
  python main.py evaluate --ckpt outputs/model_full.pt --config configs/default.yaml
  
  # Generate explanations
  python main.py explain --ckpt outputs/model_full.pt --pairs data/pairs_eval.csv
  
  # Download PDB files
  python main.py download-pdb --file data/pdb_ids.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Data command
    data_parser = subparsers.add_parser('data', help='Data management')
    data_parser.add_argument('--generate-samples', action='store_true',
                            help='Generate sample data for testing')
    data_parser.add_argument('--load-alternate', type=str,
                            help='Load and index alternate data path')
    data_parser.add_argument('--liganded-only', action='store_true',
                            help='Filter to liganded complexes only')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--config', type=str, default='configs/default.yaml',
                             help='Path to config file')
    train_parser.add_argument('--model-type', type=str, default='full',
                             choices=['full', 'classical'],
                             help='Model type')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    train_parser.add_argument('--data-path', type=str,
                             help='Override alternate_data_path from config')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--ckpt', type=str, required=True,
                            help='Path to model checkpoint')
    eval_parser.add_argument('--config', type=str, default='configs/default.yaml',
                            help='Path to config file')
    eval_parser.add_argument('--output', type=str, default='reports/',
                            help='Output directory')
    
    # Explain command
    explain_parser = subparsers.add_parser('explain', help='Generate explanations')
    explain_parser.add_argument('--ckpt', type=str, required=True,
                               help='Path to model checkpoint')
    explain_parser.add_argument('--pairs', type=str,
                               help='CSV with drug/protein pairs')
    explain_parser.add_argument('--output', type=str, default='reports/explanations.json',
                               help='Output file')
    
    # Download PDB command
    dl_parser = subparsers.add_parser('download-pdb', help='Download PDB files')
    dl_parser.add_argument('--pdb-ids', type=str, nargs='+',
                          help='PDB IDs to download')
    dl_parser.add_argument('--file', type=str,
                          help='File with PDB IDs (one per line)')
    dl_parser.add_argument('--output-dir', type=str, default='data/pdb',
                          help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route commands
    if args.command == 'data':
        if args.generate_samples:
            from scripts.generate_sample_data import (
                create_sample_drugs_csv,
                create_sample_pdb_lists,
                create_sample_exclusion_lists,
                create_sample_ligand_lookup
            )
            print("Generating sample data...\n")
            create_sample_drugs_csv()
            create_sample_pdb_lists()
            create_sample_exclusion_lists()
            create_sample_ligand_lookup()
            print("\n✅ Done!")
        elif args.load_alternate:
            from src.data.alternate_loader import AlternateDataLoader
            from src.data.data_adapter import AlternateDataAdapter
            
            print(f"\nLoading alternate data source: {args.load_alternate}")
            loader = AlternateDataLoader(args.load_alternate)
            adapter = AlternateDataAdapter(loader)
            
            # Print summary
            adapter.print_summary()
            
            # Validate
            loader.report_validation()
            
            # Save lookups
            output_dir = Path("outputs/alternate_data")
            output_dir.mkdir(parents=True, exist_ok=True)
            adapter.save_pairs_csv(str(output_dir / "complexes.csv"))
            adapter.save_drug_lookup_csv(str(output_dir / "drug_lookup.csv"))
            adapter.save_protein_lookup_csv(str(output_dir / "protein_lookup.csv"))
            print(f"\n✅ Saved outputs to {output_dir}")
        else:
            print("Use --generate-samples or --load-alternate")
    
    elif args.command == 'train':
        from src.train import main as train_main
        sys.argv = ['train.py', '--config', args.config, '--model_type', args.model_type]
        if args.data_path:
            sys.argv.extend(['--data_path', args.data_path])
        train_main()
    
    elif args.command == 'evaluate':
        print("Evaluation not yet implemented in main CLI")
        print(f"Use: python -m src.evaluate --ckpt {args.ckpt}")
    
    elif args.command == 'explain':
        print("Explain not yet implemented in main CLI")
        print(f"Use: python -m src.explain --ckpt {args.ckpt}")
    
    elif args.command == 'download-pdb':
        from scripts.download_pdbs import download_pdb_list
        
        pdb_ids = []
        if args.pdb_ids:
            pdb_ids.extend(args.pdb_ids)
        if args.file:
            with open(args.file) as f:
                pdb_ids.extend([line.strip() for line in f if line.strip()])
        
        if not pdb_ids:
            print("Specify PDB IDs with --pdb-ids or --file")
            return 1
        
        download_pdb_list(pdb_ids, args.output_dir)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
