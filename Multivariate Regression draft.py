import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -----------------------------
# Function 1: Read TSV file
# -----------------------------
#"EOD_master_diagnoses_12022025.tsv"
def read_tsv(fileName):
    initial_df = pd.read_csv(fileName, sep='\t')
    print(initial_df.head())

    #Column names: 
    #edipi, ssn, date_of_birth, date_of_death, veteran_in_vha, diagnosis_first, diagnosis_last, classification, diagnosis
    #edipi, ssn, date_of_birth, date_of_death (some in NaN), veteran_in_vha (yes / no), diagnosis_first, diagnosis_last, classification (icd10cm or icd9cm), diagnosis

    #Create a new df that stores the row if the column value of "veteran_in_vha" equals "yes"
    df = initial_df[initial_df['veteran_in_vha'] == "yes"]
    print(df.head())

# Convert dates
#df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
#df['time of diagnosis'] = (df['diagnosis_last'] - df['diagnosis_first']).dt.days / 365.25 
# Calculate age at diagnosis
#df['age_at_diagnosis'] = (df['time of diagnosis'] - df['date_of_birth']).dt.days / 365.25

# Suicide-related ICD-10 codes of interest
    suicide_codes = [
        "R45851",  # Suicidal ideation
        "T1491",   # Suicide attempt
        "Z915"     # Personal history of self-harm
    ]


    #isin(suicide_codes) → catches exact matches like R45.851, T14.91, Z91.5.
    #str.startswith(tuple([f"X{i}" for i in range(71, 84)])) → catches all codes in the X71–X83 range (intentional self-harm).

    # Codes that start with X71–X83 (intentional self-harm)
    mask_self_harm = df['diagnosis'].str.startswith(tuple([f"X{i}" for i in range(71, 84)]))

    # Filter veterans with suicide-related codes
    df_suicide = df[
        (df['veteran_in_vha'].str.lower() == "yes") &
            (
            df['diagnosis'].isin(suicide_codes) |
            mask_self_harm
        )
    ]

    print(df_suicide.head())

# Suicide & Self‑Harm
#   R45.851 → Suicidal ideation
#   T14.91 → Suicide attempt
#   X71–X83 → Intentional self‑harm (various methods)
#   Z91.5 → Personal history of self‑harm

# Mood Disorders
#   F32.x → Major depressive disorder, single episode
#   F33.x → Major depressive disorder, recurrent
#   F34.x → Persistent mood (affective) disorders (e.g., dysthymia)
#   F39 → Unspecified mood disorder

# Anxiety & Stress Disorders
#   F41.x → Anxiety disorders
#   F43.x → Reaction to severe stress, adjustment disorders (includes PTSD)

# Schizophrenia & Psychotic Disorders
#   F20.x → Schizophrenia
#   F21 → Schizotypal disorder
#   F25.x → Schizoaffective disorders

# Substance Use Disorders
#   F10–F19 → Mental and behavioral disorders due to psychoactive substance use

# Broader psychiatric code prefixes
    psychiatric_prefixes = [
        "F32", "F33", "F34", "F39",  # Mood disorders
        "F41", "F43",                # Anxiety/stress
        "F20", "F21", "F25",         # Schizophrenia/psychotic
        "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19"  # Substance use
    ]

# Mask for suicide/self-harm
    mask_suicide = df['diagnosis_code'].isin(suicide_codes) | df['diagnosis_code'].str.startswith(tuple([f"X{i}" for i in range(71, 84)]))

# Mask for psychiatric conditions
    mask_psych = df['diagnosis_code'].str.startswith(tuple(psychiatric_prefixes))

# Combine masks and filter veterans
    df_filtered = df[
        (df['veteran_in_vha'].str.lower() == "yes") &
        (mask_suicide | mask_psych)
    ]

    print(df_filtered.head())

    df_filtered.to_csv("veterans_suicide_psych.tsv", sep="\t", index=False)
    print("Filtered dataset saved as veterans_suicide_psych.tsv")

    return df_filtered





def plot_veterans_by_category(df):
    """
    Create a bar chart showing the number of veterans by psychiatric diagnosis category.
    
    Parameters:
        df (pd.DataFrame): DataFrame with at least a 'diagnosis_code' column.
    """
    
    # Categorization function
    def categorize_code(code):
        if code in ["R45.851", "T14.91", "Z91.5"] or code.startswith(tuple([f"X{i}" for i in range(71, 84)])):
            return "Suicide/Self-harm"
        elif code.startswith(("F32", "F33", "F34", "F39")):
            return "Mood disorders"
        elif code.startswith(("F41", "F43")):
            return "Anxiety/Stress disorders"
        elif code.startswith(("F20", "F21", "F25")):
            return "Schizophrenia/Psychotic disorders"
        elif code.startswith(tuple([f"F{i}" for i in range(10, 20)])):
            return "Substance use disorders"
        else:
            return "Other psychiatric"

    # Apply categorization
    df['category'] = df['diagnosis_code'].apply(categorize_code)

    # Count by category
    category_counts = df['category'].value_counts()

    # Plot
    plt.figure(figsize=(10,6))
    category_counts.plot(kind='bar', color='skyblue', edgecolor='black')

    plt.title("Veterans by Psychiatric Diagnosis Category", fontsize=14)
    plt.xlabel("Diagnosis Category", fontsize=12)
    plt.ylabel("Number of Veterans", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# --- Example usage ---
# df_filtered = pd.read_csv("veterans_suicide_psych.tsv", sep="\t")
# plot_veterans_by_category(df_filtered)

# --- Categorize ICD-10-CM codes into broader groups ---
def categorize_code(code):
    if code in ["R45.851", "T14.91", "Z91.5"] or code.startswith(tuple([f"X{i}" for i in range(71, 84)])):
        return "Suicide/Self-harm"
    elif code.startswith(("F32", "F33", "F34", "F39")):
        return "Mood disorders"
    elif code.startswith(("F41", "F43")):
        return "Anxiety/Stress disorders"
    elif code.startswith(("F20", "F21", "F25")):
        return "Schizophrenia/Psychotic disorders"
    elif code.startswith(tuple([f"F{i}" for i in range(10, 20)])):
        return "Substance use disorders"
    else:
        return "Other psychiatric"

# Apply categorization
df_filtered['category'] = df_filtered['diagnosis_code'].apply(categorize_code)

# --- Count veterans by category ---
category_counts = df_filtered['category'].value_counts()

# --- Plot bar chart ---
plt.figure(figsize=(10,6))
category_counts.plot(kind='bar', color='skyblue', edgecolor='black')

plt.title("Veterans by Psychiatric Diagnosis Category", fontsize=14)
plt.xlabel("Diagnosis Category", fontsize=12)
plt.ylabel("Number of Veterans", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Show the chart
plt.show()




if __name__ == "__main__":




# Count frequency of each diagnosis code in the filtered dataset
code_counts = df_filtered['diagnosis_code'].value_counts()

print("Counts by diagnosis code:")
print(code_counts)

# If you want counts grouped by broader categories (e.g. suicide vs mood vs psychotic)
import numpy as np

def categorize_code(code):
    if code in ["R45.851", "T14.91", "Z91.5"] or code.startswith(tuple([f"X{i}" for i in range(71, 84)])):
        return "Suicide/Self-harm"
    elif code.startswith(("F32", "F33", "F34", "F39")):
        return "Mood disorders"
    elif code.startswith(("F41", "F43")):
        return "Anxiety/Stress disorders"
    elif code.startswith(("F20", "F21", "F25")):
        return "Schizophrenia/Psychotic disorders"
    elif code.startswith(tuple([f"F{i}" for i in range(10, 20)])):
        return "Substance use disorders"
    else:
        return "Other psychiatric"

# Apply categorization
df_filtered['category'] = df_filtered['diagnosis_code'].apply(categorize_code)

# Count by category
category_counts = df_filtered['category'].value_counts()

print("\nCounts by category:")
print(category_counts)


code_counts.to_csv("diagnosis_code_counts.tsv", sep="\t")
category_counts.to_csv("diagnosis_category_counts.tsv", sep="\t")



#Diagnosis code → categorical, needs encoding. 
# Encode diagnosis code (categorical → numeric)
#One-Hot Encoding
#Each unique diagnosis code becomes a binary column (0/1).

df = pd.DataFrame({df_suicide['diagnosis']})
#df_encoded = pd.get_dummies(df, columns=["diagnosis"], drop_first=True)
print(df_encoded)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Split data   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
print("R^2 score:", model.score(X_test, y_test))

#Interpret results
#Coefficients tell you how each diagnosis code affects age at diagnosis.

#R² score shows how well the model explains variance.

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print(coefficients)