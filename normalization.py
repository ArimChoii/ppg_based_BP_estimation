import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the CSV file
df = pd.read_csv('Ubpulse-Pulse-Wave.csv')

# Display the first few rows of the dataframe
print("Original Data:")
print(df.head())

# Assuming the second column index is 1 (since indexing starts from 0)
second_column_name = df.columns[1]

# Extract the second column
second_column_data = df[[second_column_name]]

# Initialize the StandardScaler
scaler = StandardScaler()

# Perform Gaussian normalization on the second column
df[second_column_name] = scaler.fit_transform(second_column_data)

# Display the first few rows of the normalized dataframe
print("\nNormalized Data:")
print(df.head())

# Save the normalized data back to a new CSV file
df.to_csv('KNK.csv', index=False)

