import pandas as pd
import matplotlib.pyplot as plt

# Load patient data with descriptions
filename = " " #Insert file path.
patients = pd.read_csv(filename, sep="\t")

# Count frequency of each code
code_counts = patients["diagnosis"].value_counts().head(20)

# Bar chart of top 20 codes
plt.figure(figsize=(12,6))
code_counts.plot(kind="bar")
plt.title("Top 20 Most Frequent ICD Codes")
plt.xlabel("ICD Code")
plt.ylabel("Number of Patients")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show(block=False)


desc_counts = patients["diagnosis_description"].value_counts().head(20)
plt.figure(figsize=(15,15))
desc_counts.plot(kind="bar")
plt.title("Top 20 Most Frequent Diagnoses")
plt.xlabel("Diagnosis")
plt.ylabel("Number of Patients")
plt.xticks(rotation=75)
plt.tight_layout()
plt.show(block=False)


# Classify how many have ICD 10 or ICD 9 codes 
version_counts = patients["classification"].value_counts()

plt.figure(figsize=(6,6))
version_counts.plot(kind="pie", autopct="%1.1f%%", startangle=90)
plt.title("Distribution of ICD Codes by Version")
plt.ylabel("")
plt.show(block=False)


# Create categories of the diagnoses

category_keywords = {
    "Cardiovascular": ["heart", "cardiac", "hypertension", "infarction", "angina", "arrhythmia"],
    "Psychiatric": ["depression", "anxiety", "schizophrenia", "bipolar", "ptsd", "mental"],
    "Infectious": ["infection", "influenza", "virus", "bacterial", "sepsis", "tuberculosis"],
    "Respiratory": ["asthma", "bronchitis", "pneumonia", "respiratory", "copd"],
    "Musculoskeletal": ["arthritis", "fracture", "osteoporosis", "joint", "muscle"],
    "Endocrine": ["diabetes", "thyroid", "endocrine", "hormone"],
    "Neurological": ["stroke", "epilepsy", "migraine", "neuropathy", "parkinson"],
    "Digestive": ["gastritis", "ulcer", "liver", "pancreatitis", "digestive"],
    "Injury/Poisoning": ["injury", "trauma", "burn", "poison", "wound"],
    "Other": []
}

def assign_category(description):
    if pd.isna(description):
        return "Unknown"
    desc = description.lower()
    for category, keywords in category_keywords.items():
        if any(kw in desc for kw in keywords):
            return category
    return "Other"

patients["category"] = patients["diagnosis_description"].apply(assign_category)


import matplotlib.pyplot as plt

category_counts = patients["category"].value_counts()

plt.figure(figsize=(10,6))
category_counts.plot(kind="bar", color="skyblue")
plt.title("Distribution of Diagnoses by Category")
plt.xlabel("Category")
plt.ylabel("Number of Patients")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show(block=False)


#Using official WHO structure

# Define ICD-10 chapter ranges
chapter_map = [
    ("A00","B99","Chapter I: Infectious"),
    ("C00","D49","Chapter II: Neoplasms"),
    ("D50","D89","Chapter III: Blood/Immune"),
    ("E00","E89","Chapter IV: Endocrine"),
    ("F01","F99","Chapter V: Mental/Behavioral"),
    ("G00","G99","Chapter VI: Nervous"),
    ("H00","H59","Chapter VII: Eye"),
    ("H60","H95","Chapter VIII: Ear"),
    ("I00","I99","Chapter IX: Circulatory"),
    ("J00","J99","Chapter X: Respiratory"),
    ("K00","K95","Chapter XI: Digestive"),
    ("L00","L99","Chapter XII: Skin"),
    ("M00","M99","Chapter XIII: Musculoskeletal"),
    ("N00","N99","Chapter XIV: Genitourinary"),
    ("O00","O9A","Chapter XV: Pregnancy"),
    ("P00","P96","Chapter XVI: Perinatal"),
    ("Q00","Q99","Chapter XVII: Congenital"),
    ("R00","R99","Chapter XVIII: Symptoms/Signs"),
    ("S00","T88","Chapter XIX: Injury/Poisoning"),
    ("V00","Y99","Chapter XX: External causes"),
    ("Z00","Z99","Chapter XXI: Health status"),
    ("U00","U85","Chapter XXII: Special purposes"),
]

# Function to assign chapter
def assign_chapter(code):
    if pd.isna(code): return "Unknown"
    code = code.strip().upper()
    for start, end, chapter in chapter_map:
        if start <= code <= end:
            return chapter
    return "Other"

patients["icd10_chapter"] = patients["diagnosis_code"].apply(assign_chapter)

# Count and plot
import matplotlib.pyplot as plt
chapter_counts = patients["icd10_chapter"].value_counts()

chapter_counts.plot(kind="bar", color="steelblue")
plt.title("Distribution of Diagnoses by ICD-10 Chapter")
plt.xlabel("ICD-10 Chapter")
plt.ylabel("Number of Patients")
plt.xticks(rotation=75)
plt.tight_layout()
plt.show(block=False)

