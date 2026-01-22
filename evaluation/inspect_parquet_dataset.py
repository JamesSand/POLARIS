import os
import pandas as pd

# Path to the benchmarks folder
# benchmarks_folder = './benchmarks'
benchmarks_folder = './benchmarks/processed'

# Get the directory name for output file
dir_name = os.path.basename(os.path.normpath(benchmarks_folder))

# Collect all output lines
all_output_lines = []

# Iterate through each parquet file in the benchmarks folder
for filename in os.listdir(benchmarks_folder):
    if filename.endswith('.parquet'):
        # Load the parquet file
        file_path = os.path.join(benchmarks_folder, filename)
        df = pd.read_parquet(file_path)

        # print(f'Inspecting {filename}...')
        # print(len(df), 'samples found.')
        # continue

        # Get the first sample
        first_sample = df.iloc[0]

        # Prepare the output
        all_output_lines.append(f'\n{"="*80}')
        all_output_lines.append(f'Properties of {filename}')
        all_output_lines.append(f'{"="*80}')

        # Iterate through each attribute in the first sample
        for attribute, value in first_sample.items():
            if isinstance(value, dict):
                # If the value is a dict, print its keys and values
                for key, val in value.items():
                    all_output_lines.append(f'{attribute}.{key}: {val} (type: {type(val).__name__})')
            else:
                all_output_lines.append(f'{attribute}: {value} (type: {type(value).__name__})')

        print(f'Processed {filename}')

# Save all output to a single txt file in current directory
output_file_path = f'{dir_name}.txt'
with open(output_file_path, 'w') as output_file:
    output_file.write('\n'.join(all_output_lines))

print(f'All results saved to {output_file_path}')