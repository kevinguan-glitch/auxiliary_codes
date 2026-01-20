import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -----------------------------
# Function 1: Read TSV file
# -----------------------------
def read_tsv(filepath):
    """
    Reads a TSV file into a pandas DataFrame.
    """
    df = pd.read_csv(filepath, sep="\t")
    return df


# -----------------------------
# Helper: Categorize ICD-10 codes
# -----------------------------
def categorize_code(code):
    if pd.isna(code):
        return "Unknown"
    if code in ["R45.851", "T14.91", "Z91.5"] or str(code).startswith(tuple([f"X{i}" for i in range(71, 84)])):
        return "Suicide/Self-harm"
    elif str(code).startswith(("F32", "F33", "F34", "F39")):
        return "Mood disorders"
    elif str(code).startswith(("F41", "F43")):
        return "Anxiety/Stress disorders"
    elif str(code).startswith(("F20", "F21", "F25")):
        return "Schizophrenia/Psychotic disorders"
    elif str(code).startswith(tuple([f"F{i}" for i in range(10, 20)])):
        return "Substance use disorders"
    else:
        return "Other psychiatric"
    
# -----------------------------
# Function 2: Generate Pie Chart
# -----------------------------
def plot_pie_chart(df):
    """
    Generates a pie chart of diagnosis categories.
    """
    df['category'] = df['diagnosis'].apply(categorize_code)
    category_counts = df['category'].value_counts()

    plt.figure(figsize=(8,8))
    category_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title("Veterans by Psychiatric Diagnosis Category")
    plt.ylabel("")  # Hide y-label
    plt.show()


# -----------------------------
# Function 3: Multivariate Regression
# -----------------------------
def run_regression(df):
    """
    Runs a multivariate regression predicting age at diagnosis from diagnosis codes.
    """
    # Convert dates
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
    df['date_of_diagnosis'] = pd.to_datetime(df['date_of_diagnosis'], errors='coerce')

    # Calculate age at diagnosis
    df['age_at_diagnosis'] = (df['date_of_diagnosis'] - df['date_of_birth']).dt.days / 365.25

    # Encode diagnosis codes (one-hot)
    df_encoded = pd.get_dummies(df, columns=['diagnosis'], drop_first=True)

    # Drop unused columns
    df_encoded = df_encoded.drop(columns=['ID', 'date_of_birth', 'date_of_diagnosis'], errors='ignore')

    # Define features and target
    X = df_encoded.drop(columns=['age_at_diagnosis'])
    y = df_encoded['age_at_diagnosis']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    print("R^2 score:", model.score(X_test, y_test))
    print("Coefficients:")
    print(pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_}))

    return model

# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    # Step 1: Read data
    df = read_tsv(" ")

    # Step 2: Generate pie chart
    plot_pie_chart(df)

    # Step 3: Run regression
    model = run_regression(df)