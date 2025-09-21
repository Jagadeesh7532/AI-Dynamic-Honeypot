import json
import pandas as pd
import os

# Path to Cowrie JSON log file
log_file = '/home/tillu/Desktop/cowrie/var/log/cowrie/cowrie.json'

# Path to output CSV file
output_csv = 'attack_logs.csv'

# Check if log file exists
if not os.path.isfile(log_file):
    print(f'Log file {log_file} does not exist.')
    exit()

# Read JSON log file line by line and parse
log_entries = []
with open(log_file, 'r') as f:
    for line in f:
        log_entry = json.loads(line)
        log_entries.append(log_entry)

# Convert to pandas DataFrame
df = pd.DataFrame(log_entries)

# Save DataFrame to CSV
df.to_csv(output_csv, index=False)
print(f'Parsed logs saved to {output_csv}')

