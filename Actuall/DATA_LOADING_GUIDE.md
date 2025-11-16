# Data Loading and Cleaning Guide

## 📊 Understanding Data Loading

### Current Implementation

The `DrugPatientDataProcessor` loads data in several steps. Let me explain what's happening:

## 1️⃣ Drug Data Loading

### What Files Are Loaded?

```python
# Search pattern
pattern = "{data_dir}/**/results/*_descriptors_3d.csv"

# Example: /media/priyanshu/SD/othercode/data/**/results/*_descriptors_3d.csv
```

### Loading Process

```python
processor.load_protein_ligand_data(max_samples=None)  # Load ALL by default
```

**Step-by-step**:

1. **Search for CSV files**
   ```python
   # Finds all files matching pattern
   csv_files = glob.glob(pattern, recursive=True)
   # Example: Found 5,234 descriptor files
   ```

2. **Optional Filtering** (if max_samples is set)
   ```python
   # In notebook:
   MAX_DRUGS = 100  # ⚠️ THIS LIMITS DATA!
   processor.load_protein_ligand_data(max_samples=MAX_DRUGS)
   # Only loads first 100 files out of 5,234 found

   # To load ALL:
   MAX_DRUGS = None  # ✅ Loads everything
   processor.load_protein_ligand_data(max_samples=None)
   ```

3. **For each CSV file**:
   - Read CSV
   - Extract descriptor columns
   - Handle missing values
   - Add to graph

## 2️⃣ Data Cleaning Steps

### A. Column Selection

```python
# Expected columns (from your PPI dataset)
descriptor_cols = [
    'Volume', 'PMI1', 'PMI2', 'PMI3',  # Shape descriptors
    'NPR1', 'NPR2', 'Rgyr',             # Size descriptors
    'Asphericity', 'SpherocityIndex',   # Geometry
    'Eccentricity', 'InertialShapeFactor'
]

atom_cols = ['CZ', 'CA', 'O', 'OD1', 'OG', 'N', 'NZ', 'DU']

# Only uses columns that exist in the CSV
available_cols = [col for col in all_cols if col in df.columns]
```

**What this means**:
- ✅ Flexible: Works even if some columns are missing
- ✅ No data loss: Uses all available valid columns
- ⚠️ Skips files with NO valid columns

### B. Missing Value Handling

```python
# Extract features
features = df[available_cols].iloc[0].values.astype(np.float32)

# Replace NaN with 0
features = np.nan_to_num(features, nan=0.0)
```

**Cleaning strategy**:
- `NaN` → `0.0`
- `Inf` → Large finite number
- `-Inf` → Large negative number

**Impact**:
- ✅ No data rows are dropped
- ⚠️ Missing values imputed as 0 (may affect model)

### C. Error Handling

```python
try:
    # Load and process file
    ...
except Exception as e:
    print(f"Warning: Failed to load {csv_file}: {e}")
    continue  # Skip this file, continue with next
```

**Files skipped**:
- Corrupted CSV files
- Files without valid columns
- Permission errors
- Invalid file formats

## 3️⃣ Data Loss Analysis

### Where Data Might Be Cut:

#### ❌ **Issue 1: max_samples Parameter**

```python
# In notebooks, this limits data:
MAX_DRUGS = 100  # ⚠️ Only loads 100 drugs

# Solution: Set to None
MAX_DRUGS = None  # ✅ Loads ALL drugs
```

#### ❌ **Issue 2: File Loading Errors**

Some files might fail to load:
- Corrupted files
- Missing columns
- Invalid data types

**Check how many failed**:
```python
csv_files_found = glob.glob(pattern, recursive=True)
print(f"Total files found: {len(csv_files_found)}")

drugs_loaded = processor.load_protein_ligand_data(max_samples=None)
print(f"Drugs loaded: {drugs_loaded}")
print(f"Failed/Skipped: {len(csv_files_found) - drugs_loaded}")
```

#### ❌ **Issue 3: Multiple Rows in CSV**

```python
# Only uses FIRST row
features = df[available_cols].iloc[0].values
```

**Why**: Each CSV typically has one descriptor set per protein pocket.

If your CSV has multiple rows, only the first is used. ⚠️

## 4️⃣ Recommended: Load ALL Data

### Update Notebooks to Load Everything

**In `train_model.ipynb`**:
```python
# Change this:
MAX_DRUGS = 100  # ❌ Limited

# To this:
MAX_DRUGS = None  # ✅ Load all available drugs
```

**In `train_model_spark.ipynb`**:
```python
# Change this:
MAX_DRUGS = 200  # ❌ Limited

# To this:
MAX_DRUGS = None  # ✅ Load all
```

## 5️⃣ Enhanced Data Loading (Recommended)

Let me create an enhanced version that gives you full visibility:

```python
def load_protein_ligand_data_verbose(self, max_samples=None,
                                     skip_invalid=True,
                                     impute_strategy='zero'):
    """
    Enhanced data loading with detailed logging.

    Args:
        max_samples: Max drugs to load (None = all)
        skip_invalid: Skip files with errors (vs raise exception)
        impute_strategy: 'zero', 'mean', 'median', or 'drop'
    """
    pattern = os.path.join(self.data_dir, "**", "results", "*_descriptors_3d.csv")
    csv_files = glob.glob(pattern, recursive=True)

    print(f"=" * 70)
    print(f"DATA LOADING REPORT")
    print(f"=" * 70)
    print(f"Files found: {len(csv_files)}")
    print(f"Max samples: {max_samples if max_samples else 'ALL'}")
    print(f"Impute strategy: {impute_strategy}")
    print(f"=" * 70)

    stats = {
        'files_found': len(csv_files),
        'files_processed': 0,
        'files_skipped': 0,
        'missing_columns': 0,
        'missing_values': 0,
        'errors': []
    }

    files_to_process = csv_files[:max_samples] if max_samples else csv_files

    for idx, csv_file in enumerate(files_to_process):
        if (idx + 1) % 1000 == 0:
            print(f"Progress: {idx + 1}/{len(files_to_process)} files processed...")

        try:
            df = pd.read_csv(csv_file)

            # Expected columns
            descriptor_cols = [
                'Volume', 'PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2',
                'Rgyr', 'Asphericity', 'SpherocityIndex', 'Eccentricity',
                'InertialShapeFactor'
            ]
            atom_cols = ['CZ', 'CA', 'O', 'OD1', 'OG', 'N', 'NZ', 'DU']
            all_cols = descriptor_cols + atom_cols

            # Check available columns
            available_cols = [col for col in all_cols if col in df.columns]
            missing_cols = set(all_cols) - set(available_cols)

            if len(available_cols) == 0:
                stats['files_skipped'] += 1
                stats['errors'].append(f"No valid columns: {csv_file}")
                continue

            if len(missing_cols) > 0:
                stats['missing_columns'] += 1

            # Extract features
            features = df[available_cols].iloc[0].values.astype(np.float32)

            # Check for missing values
            if np.any(np.isnan(features)):
                stats['missing_values'] += 1

                # Imputation
                if impute_strategy == 'zero':
                    features = np.nan_to_num(features, nan=0.0)
                elif impute_strategy == 'mean':
                    col_means = df[available_cols].mean()
                    features = np.where(np.isnan(features), col_means, features)
                elif impute_strategy == 'median':
                    col_medians = df[available_cols].median()
                    features = np.where(np.isnan(features), col_medians, features)
                elif impute_strategy == 'drop':
                    stats['files_skipped'] += 1
                    continue

            # Create drug ID
            drug_id = Path(csv_file).stem

            # Add to graph
            self.graph.add_drug(drug_id, features)
            stats['files_processed'] += 1

        except Exception as e:
            stats['files_skipped'] += 1
            stats['errors'].append(f"{csv_file}: {str(e)}")
            if not skip_invalid:
                raise

    # Print summary
    print(f"\n" + "=" * 70)
    print(f"LOADING SUMMARY")
    print(f"=" * 70)
    print(f"Files found:          {stats['files_found']}")
    print(f"Files processed:      {stats['files_processed']}")
    print(f"Files skipped:        {stats['files_skipped']}")
    print(f"Missing columns:      {stats['missing_columns']}")
    print(f"Files with NaN:       {stats['missing_values']}")
    print(f"Errors:               {len(stats['errors'])}")

    if len(stats['errors']) > 0:
        print(f"\nFirst 5 errors:")
        for err in stats['errors'][:5]:
            print(f"  - {err}")

    print(f"=" * 70)

    return stats['files_processed']
```

## 6️⃣ Data Quality Checks

### Add This Cell to Your Notebook

```python
# After loading data, run quality checks

import glob

# 1. Check how many files exist
pattern = "/media/priyanshu/SD/othercode/data/**/results/*_descriptors_3d.csv"
csv_files = glob.glob(pattern, recursive=True)
print(f"Total descriptor files available: {len(csv_files)}")

# 2. Load data
processor = DrugPatientDataProcessor(
    data_dir="/media/priyanshu/SD/othercode/data"
)
drugs_loaded = processor.load_protein_ligand_data(max_samples=None)

# 3. Compare
print(f"\nData Loading Report:")
print(f"Files found:     {len(csv_files)}")
print(f"Drugs loaded:    {drugs_loaded}")
print(f"Success rate:    {drugs_loaded/len(csv_files)*100:.1f}%")
print(f"Failed/Skipped:  {len(csv_files) - drugs_loaded}")

# 4. Get statistics
stats = processor.get_statistics()
print(f"\nGraph Statistics:")
for key, val in stats.items():
    print(f"  {key}: {val}")
```

## 7️⃣ Interaction Data

### Not Cut by Default

```python
# Creates interactions based on rate
processor.create_synthetic_interactions(interaction_rate=0.05)

# This creates 5% of possible drug-patient pairs
# Example: 100 drugs × 200 patients = 20,000 possible
#          × 0.05 = 1,000 interactions created
```

**To use all combinations** (not recommended for large datasets):
```python
interaction_rate = 1.0  # Create ALL possible pairs
# Warning: 1000 drugs × 5000 patients = 5 million edges!
```

## 8️⃣ Recommendations

### For Complete Data Loading:

1. **Set max_samples = None**
   ```python
   MAX_DRUGS = None  # In notebook config
   ```

2. **Check loading statistics**
   ```python
   # Add after loading
   print(f"Files found: {len(csv_files)}")
   print(f"Drugs loaded: {processor.graph.num_drugs()}")
   ```

3. **Monitor for errors**
   - Check notebook output for "Warning: Failed to load" messages
   - Count how many files were skipped

4. **Validate data quality**
   ```python
   # Check for reasonable feature values
   drug_features = processor.graph.get_drug_features_matrix()
   print(f"Feature stats:")
   print(f"  Mean: {drug_features.mean(axis=0)}")
   print(f"  Std:  {drug_features.std(axis=0)}")
   print(f"  Min:  {drug_features.min(axis=0)}")
   print(f"  Max:  {drug_features.max(axis=0)}")
   ```

## 9️⃣ Summary

### Current Cleaning Strategy:

✅ **What's Good**:
- Flexible column matching
- Handles missing files gracefully
- NaN imputation (no row drops)
- Error recovery (continues on failure)

⚠️ **What to Watch**:
- `max_samples` parameter limits data
- Only first row of CSV used
- NaN imputed as 0 (may not be ideal)
- Silent failures (files skipped without notice)

### To Load 100% of Available Data:

1. Set `MAX_DRUGS = None`
2. Set `interaction_rate` appropriately
3. Check loading logs
4. Validate statistics

---

**Bottom Line**: By default, the code tries to load ALL data, but **notebook configurations may limit it**. Always check `MAX_DRUGS` parameter!
