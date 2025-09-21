import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load attack logs CSV
df_logs = pd.read_csv('attack_logs.csv')

# Data cleaning
df_logs.dropna(subset=['eventid', 'timestamp', 'session', 'src_ip'], inplace=True)

# Convert timestamp to datetime
df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])

# Filter command input events
commands = df_logs[df_logs['eventid'] == 'cowrie.command.input']

# Feature: Command count per session
command_counts = commands.groupby('session')['input'].count().reset_index()
command_counts.rename(columns={'input': 'command_count'}, inplace=True)

# Feature: Session duration
session_times = df_logs.groupby('session')['timestamp'].agg(['min', 'max']).reset_index()
session_times['session_duration'] = (session_times['max'] - session_times['min']).dt.total_seconds()

# Merge features
features_df = command_counts.merge(session_times[['session', 'session_duration']], on='session')
features_df = features_df.merge(df_logs[['session', 'src_ip']].drop_duplicates(), on='session')

# Label data (assume command_count > 20 is suspicious)
features_df['label'] = features_df['command_count'].apply(lambda x: 1 if x > 20 else 0)

# Select features and target
X = features_df[['command_count', 'session_duration']]
y = features_df['label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')

# Save processed data for training
processed_data = pd.DataFrame(X_scaled, columns=['command_count_scaled', 'session_duration_scaled'])
processed_data['label'] = y.reset_index(drop=True)
print(processed_data.head())  # Print the first few rows for debugging
processed_data.to_csv('processed_data.csv', index=False)

print('Data preprocessing completed. Processed data saved to processed_data.csv')

