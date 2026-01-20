import pandas as pd

# --- Load ICD-9-CM (Excel file) ---
icd9 = pd.read_excel("icd9cm_codes.xlsx", sheet_name=0)
icd9.columns = ["code", "description", "short description"]  # normalize column names

# --- Load ICD-10-CM (Text file) ---
#the ICD‑10 TXT file isn’t tab‑delimited into separate columns, it’s just one string per row

# Read ICD-10 file as raw lines
with open("icd10cm_codes.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Split each line into code and description
records = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    # Split on first whitespace only
    parts = line.split(maxsplit=1)
    if len(parts) == 2:
        code, desc = parts
    else:
        code, desc = parts[0], ""
    records.append((code, desc))

# Convert to DataFrame
icd10 = pd.DataFrame(records, columns=["code", "description"])

icd9["version"] = "ICD9" 
icd10["version"] = "ICD10"
# --- Combine both ---
merged = pd.concat([icd9, icd10], ignore_index=True)

# Drop duplicates if same code appears in both
merged = merged.drop_duplicates(subset="code")

# --- Save to CSV ---
merged.to_csv("merged_icd_codes.csv", index=False)

