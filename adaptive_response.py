import pandas as pd
import joblib
import subprocess
from datetime import datetime, timedelta, timezone
import os
import json

# Load model and scaler
model_path = '/home/tillu/Desktop/cowrie/models/rf_model.pkl'
scaler_path = '/home/tillu/Desktop/cowrie/models/scaler.pkl'

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    print('Model or scaler files not found.')
    exit()

print('Loading model and scaler...')
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
print('Model and scaler loaded successfully.')

# Paths
log_file = '/home/tillu/Desktop/cowrie/var/log/cowrie/cowrie.json'
cowrie_config_file = '/home/tillu/Desktop/cowrie/etc/cowrie.cfg'
cowrie_directory = '/home/tillu/Desktop/cowrie'

def adapt_honeypot():
    """Adapt the honeypot configuration by changing the SSH port to 2022 and restarting Cowrie."""
    print('Adapting honeypot configuration...')
    # Change SSH port to 2022
    with open(cowrie_config_file, 'r') as file:
        config_lines = file.readlines()

    with open(cowrie_config_file, 'w') as file:
        for line in config_lines:
            if line.strip().startswith('listen_port'):
                file.write('listen_port = 2022\n')
            else:
                file.write(line)

    # Restart Cowrie
    try:
        # Use Cowrie's control script
        subprocess.run(['bin/cowrie', 'stop'], cwd=cowrie_directory, check=True)
        subprocess.run(['bin/cowrie', 'start'], cwd=cowrie_directory, check=True)
        print('Cowrie restarted successfully.')
    except subprocess.CalledProcessError as e:
        print(f"Failed to restart Cowrie: {e}")

def monitor_and_adapt():
    """Monitor Cowrie logs from the last hour, preprocess the data, and adapt the honeypot if suspicious activity is detected."""
    print('Starting monitoring process...')
    # Check if log file exists
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    print('Parsing logs from the last hour...')
    # Initialize an empty list for log entries
    log_entries = []

    # Load logs from the last hour
    current_time = datetime.now(timezone.utc)
    one_hour_ago = current_time - timedelta(hours=1)

    # Parse logs
    with open(log_file, 'r') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                log_entry = json.loads(line)
                # Handle timestamps with or without microseconds
                timestamp_str = log_entry.get('timestamp')
                if timestamp_str is None:
                    continue
                try:
                    timestamp = datetime.strptime(
                        timestamp_str, '%Y-%m-%dT%H:%M:%S.%fZ'
                    ).replace(tzinfo=timezone.utc)
                except ValueError:
                    try:
                        timestamp = datetime.strptime(
                            timestamp_str, '%Y-%m-%dT%H:%M:%SZ'
                        ).replace(tzinfo=timezone.utc)
                    except ValueError:
                        # If timestamp format does not match, skip the line
                        print(f"Invalid timestamp format in line {line_number}. Skipping line.")
                        continue
                if timestamp >= one_hour_ago:
                    log_entries.append(log_entry)
            except json.JSONDecodeError as e:
                # Ignore incomplete or corrupted lines
                print(f"Error decoding JSON in line {line_number}: {e}. Skipping line.")
            except Exception as e:
                print(f"Unexpected error in line {line_number}: {e}. Skipping line.")

    # Now, log_entries is guaranteed to be defined
    if not log_entries:
        print('No recent logs to analyze.')
        return

    print(f'Found {len(log_entries)} log entries.')

    df = pd.DataFrame(log_entries)

    # Data preprocessing
    required_columns = ['eventid', 'session', 'timestamp']
    if not all(column in df.columns for column in required_columns):
        print('Required columns not found in logs.')
        return

    df.dropna(subset=required_columns, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Filter for command input events
    commands = df[df['eventid'] == 'cowrie.command.input']

    if commands.empty:
        print('No command input events found in logs.')
        return

    # Count commands per session
    command_counts = commands.groupby('session')['input'].count().reset_index()
    command_counts.rename(columns={'input': 'command_count'}, inplace=True)

    # Calculate session duration
    session_times = df.groupby('session')['timestamp'].agg(['min', 'max']).reset_index()
    session_times['session_duration'] = (session_times['max'] - session_times['min']).dt.total_seconds()

    # Merge command counts and session durations
    features_df = command_counts.merge(session_times[['session', 'session_duration']], on='session')

    if features_df.empty:
        print('No data to analyze after preprocessing.')
        return

    # Prepare features for prediction
    X_new = features_df[['command_count', 'session_duration']]

    # Ensure correct order of features
    X_new = X_new[['command_count', 'session_duration']]

    X_new_scaled = scaler.transform(X_new)

    # Predict
    predictions = model.predict(X_new_scaled)

    if any(predictions):
        print('Suspicious activity detected. Sessions:')
        suspicious_sessions = features_df[predictions == 1]['session']
        print(suspicious_sessions.tolist())
        adapt_honeypot()
    else:
        print('No suspicious activity detected.')

if __name__ == '__main__':
    monitor_and_adapt()
