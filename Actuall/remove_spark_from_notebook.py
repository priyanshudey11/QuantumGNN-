import json

nb_path = '/home/priyanshu/QuantumGNN-/Actuall/train_model_quantum.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

new_cells = []

for cell in nb['cells']:
    # Clear all outputs
    if 'outputs' in cell:
        cell['outputs'] = []
    if 'execution_count' in cell:
        cell['execution_count'] = None

    source = ''.join(cell['source'])
    
    # Skip Spark-specific cells
    if 'Spark Configuration' in source:
        # Keep config but remove Spark parts
        new_source = []
        for line in cell['source']:
            if 'SPARK' in line or 'PARTITION' in line or 'EXECUTOR' in line:
                continue
            new_source.append(line)
        cell['source'] = new_source
        new_cells.append(cell)
    elif 'Setup Spark' in source:
        # Change header
        cell['source'] = ["## Setup Imports\n"]
        new_cells.append(cell)
    elif 'pyspark' in source or 'SparkSession' in source:
        # Remove Spark imports and setup
        new_source = []
        for line in cell['source']:
            if 'pyspark' in line or 'SparkSession' in line or 'SparkContext' in line or 'PYSPARK_AVAILABLE' in line:
                continue
            if 'print("Spark Session Initialized")' in line:
                continue
            if 'spark.' in line:
                continue
            new_source.append(line)
        cell['source'] = new_source
        new_cells.append(cell)
    elif 'Loading Data with Spark' in source:
        # Change print
        cell['source'] = [line.replace('Loading Data with Spark', 'Loading Data') for line in cell['source']]
        new_cells.append(cell)
    elif 'Creating Spark DataFrames' in source:
        # Remove Spark DataFrame creation
        new_source = []
        skip = False
        for line in cell['source']:
            if 'Creating Spark DataFrames' in line:
                new_source.append('print("Creating DataFrames")\n')
            elif 'Convert to Spark DataFrame' in line:
                skip = True
            elif 'df_spark =' in line:
                continue
            elif 'df_spark.' in line:
                continue
            elif 'PARTITION_SIZE' in line:
                continue
            elif 'Schema:' in line or 'Sample data:' in line:
                continue
            elif skip and line.strip() == '':
                skip = False
            elif not skip:
                new_source.append(line)
        cell['source'] = new_source
        new_cells.append(cell)
    elif 'Convert to Spark DataFrame' in source:
        # Remove this block entirely from split cell
        new_source = []
        for line in cell['source']:
            if 'Convert to Spark DataFrame' in line:
                break # Stop adding lines from this cell
            new_source.append(line)
        # Add the rest of the cell after the spark part if any (but usually it's at the end)
        # Actually, let's just filter out the spark lines
        filtered_source = []
        for line in cell['source']:
            if 'spark.createDataFrame' in line or 'repartition' in line or 'cache()' in line:
                continue
            filtered_source.append(line)
        cell['source'] = filtered_source
        new_cells.append(cell)
    elif 'Classical mode - fully compatible with Spark' in source:
         # Update print statement
        cell['source'] = [line.replace('Classical mode - fully compatible with Spark distribution', 'Classical mode') for line in cell['source']]
        new_cells.append(cell)
    elif 'Starting distributed training' in source:
         cell['source'] = [line.replace('Starting distributed training', 'Starting training') for line in cell['source']]
         new_cells.append(cell)
    else:
        new_cells.append(cell)

nb['cells'] = new_cells

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Removed Spark dependencies from notebook.")
