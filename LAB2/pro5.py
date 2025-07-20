import pandas as pd

# Load the data
file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB2\Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

# Filter only binary columns (0/1 or True/False)
binary_cols = []
for col in df.columns:
    unique_vals = df[col].dropna().unique()
    if set(unique_vals).issubset({0, 1}):
        binary_cols.append(col)

print(f"\nBinary Columns Used: {binary_cols}")

# Take first 2 rows and select only binary attributes
v1 = df.loc[0, binary_cols].astype(int).values
v2 = df.loc[1, binary_cols].astype(int).values

# Initialize counters
f11 = f10 = f01 = f00 = 0

for i in range(len(v1)):
    if v1[i] == 1 and v2[i] == 1:
        f11 += 1
    elif v1[i] == 1 and v2[i] == 0:
        f10 += 1
    elif v1[i] == 0 and v2[i] == 1:
        f01 += 1
    elif v1[i] == 0 and v2[i] == 0:
        f00 += 1

# Calculate JC and SMC
jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) != 0 else 0
smc = (f11 + f00) / (f11 + f10 + f01 + f00) if (f11 + f10 + f01 + f00) != 0 else 0

# Display Results
print(f"\nf11 = {f11}, f10 = {f10}, f01 = {f01}, f00 = {f00}")
print(f"Jaccard Coefficient (JC): {jc:.4f}")
print(f"Simple Matching Coefficient (SMC): {smc:.4f}")
