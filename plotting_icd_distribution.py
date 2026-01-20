import pandas as pd

# Load TSV file
patients = pd.read_csv(" ", sep="\t") #insert file path 

# Inspect the first few rows
print(patients.head())

# Keep only codes starting with letters C through N
filtered = patients[patients["norm_diagnosis"].str.match(r"^[C-N]")].copy()

# Count frequency of each ICD code
code_counts = filtered["norm_diagnosis"].value_counts()
print(code_counts.head(10))

codes = patients["norm_diagnosis"].astype(str) 
# --- 2. Define ICD-10 chapter ranges 

chapter_map = { 
    "C": "Neoplasms", 
    "D": "Blood & immune disorders", 
    "E": "Endocrine, nutritional, metabolic", 
    "F": "Mental & behavioral disorders", 
    "G": "Nervous system", 
    "H": "Eye & ear diseases", 
    "I": "Circulatory system", 
    "J": "Respiratory system", 
    "K": "Digestive system", 
    "L": "Skin & subcutaneous tissue", 
    "M": "Musculoskeletal & connective tissue", 
    "N": "Genitourinary system" 
}

# --- 4. Map each code to its chapter 
def map_chapter(code: str) -> str: 
    if isinstance(code, str) and len(code) > 0: 
        return chapter_map.get(code[0], "Other") 
    return "Other"

filtered["ICD_chapter"] = filtered["norm_diagnosis"].apply(map_chapter)

chapter_counts = filtered["ICD_chapter"].value_counts()

import matplotlib.pyplot as plt

# Bar chart of top 20 ICD codes
plt.figure(figsize=(12,6))
code_counts.head(20).plot(kind="bar")
plt.title("Number of ICD Codes (C–N)")
plt.xlabel("ICD Codes")
plt.ylabel("Number of Patients")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

desc_counts = filtered["diagnosis_description"].value_counts().head(20)

plt.figure(figsize=(12,6))
desc_counts.plot(kind="bar")
plt.title("Top 20 Diagnoses (C–N)")
plt.xlabel("Diagnosis")
plt.ylabel("Number of Patients")
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()

# --- Count patients per chapter ---
chapter_counts = filtered["ICD_chapter"].value_counts() 
print(chapter_counts)

chapter_counts.plot(kind="bar", figsize=(10,6), color="skyblue", edgecolor="black")
plt.title("Patient Distribution by ICD-10 Chapter (C–N inclusive)")
plt.xlabel("ICD-10 Chapter")
plt.ylabel("Number of Patients")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

