import os
import glob
import pandas as pd
import sys
from tqdm import tqdm

def generate_report(data_dir, output_file="DATA_REPORT.md"):
    print(f"🚀 Starting Data Report Generation...")
    print(f"📂 Data Directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: Directory not found: {data_dir}")
        return

    # Expected columns
    expected_descriptor_cols = [
        'Volume', 'PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2',
        'Rgyr', 'Asphericity', 'SpherocityIndex', 'Eccentricity',
        'InertialShapeFactor'
    ]
    
    # Search for files
    pattern = os.path.join(data_dir, "**", "results", "**", "*_descriptors_3d.csv")
    print(f"🔎 Scanning for files...")
    csv_files = glob.glob(pattern, recursive=True)
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith('.')]
    
    total_files = len(csv_files)
    print(f"✅ Found {total_files} files.")
    
    if total_files == 0:
        print("❌ No files found to report on.")
        return

    # Validation stats
    valid_files = []
    invalid_files = []
    error_types = {}

    print("🧪 Validating files...")
    for csv_file in tqdm(csv_files):
        try:
            df = pd.read_csv(csv_file)
            columns = df.columns.tolist()
            
            # Check for descriptors
            missing = [col for col in expected_descriptor_cols if col not in columns]
            
            if not missing:
                valid_files.append(csv_file)
            else:
                # Determine error type
                if "Warning" in str(columns) or "Error" in str(columns):
                    reason = "Contains Error Message"
                elif len(columns) < 2:
                    reason = "Empty/Corrupt File"
                else:
                    reason = "Missing Columns"
                
                invalid_files.append({
                    "path": csv_file,
                    "reason": reason,
                    "columns_found": str(columns[:5]) + "..." if len(columns) > 5 else str(columns)
                })
                
                error_types[reason] = error_types.get(reason, 0) + 1
                
        except Exception as e:
            invalid_files.append({
                "path": csv_file,
                "reason": f"Read Error: {str(e)}",
                "columns_found": "N/A"
            })
            error_types["Read Error"] = error_types.get("Read Error", 0) + 1

    # Generate Markdown Report
    print(f"📝 Writing report to {output_file}...")
    
    with open(output_file, "w") as f:
        f.write("# 📊 Data Health Report\n\n")
        f.write(f"**Date:** {pd.Timestamp.now()}\n")
        f.write(f"**Data Directory:** `{data_dir}`\n\n")
        
        f.write("## 📈 Executive Summary\n\n")
        f.write(f"- **Total Files Scanned:** {total_files}\n")
        f.write(f"- **✅ Valid Files:** {len(valid_files)} ({len(valid_files)/total_files*100:.1f}%)\n")
        f.write(f"- **❌ Invalid Files:** {len(invalid_files)} ({len(invalid_files)/total_files*100:.1f}%)\n\n")
        
        f.write("### Error Distribution\n")
        if error_types:
            for error, count in error_types.items():
                f.write(f"- **{error}:** {count} files\n")
        else:
            f.write("- No errors found! 🎉\n")
            
        f.write("\n---\n\n")
        
        f.write("## ❌ Invalid Files Log\n\n")
        if invalid_files:
            f.write("| File Path | Error Reason | Columns Found (First 5) |\n")
            f.write("|---|---|---|\n")
            # Limit to first 100 errors to keep file size manageable
            for item in invalid_files[:100]:
                rel_path = os.path.relpath(item['path'], data_dir)
                f.write(f"| `{rel_path}` | {item['reason']} | `{item['columns_found']}` |\n")
            
            if len(invalid_files) > 100:
                f.write(f"\n*(... and {len(invalid_files) - 100} more invalid files)*\n")
        else:
            f.write("No invalid files found.\n")
            
        f.write("\n---\n\n")
        
        f.write("## ✅ Valid Files Preview\n\n")
        if valid_files:
            f.write("Here are a few examples of valid files that match the schema:\n\n")
            for path in valid_files[:10]:
                rel_path = os.path.relpath(path, data_dir)
                f.write(f"- `{rel_path}`\n")
        else:
            f.write("No valid files found.\n")

    print(f"✨ Report generated successfully: {output_file}")

if __name__ == "__main__":
    # Default path
    DEFAULT_DATA_DIR = "/media/priyanshu/SD/othercode/data"
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = DEFAULT_DATA_DIR
        
    generate_report(data_dir)
