import pandas as pd

# Input and output file paths
input_file = 'kino_2025_04.xlsx'  # Replace with your input file name
output_file = 'output.csv'  # Replace with your desired output file name

# Read the Excel file, skipping the first 3 rows
df = pd.read_excel(input_file, skiprows=3)

# Keep only columns D to W (columns are 0-indexed, so D=3, W=22)
df = df.iloc[:, 3:23]

# Save the result as a CSV file
df.to_csv(output_file, index=False)

print(f"File processed and saved as {output_file}")