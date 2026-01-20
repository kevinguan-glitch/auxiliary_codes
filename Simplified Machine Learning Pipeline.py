# Step 1: Group diagnosis codes
def group_diagnosis(code):
    return code[:3]  # Simple prefix-based grouping

# Step 2: Feature engineering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Example data
diagnosis_codes = ["53081", "V0481", "F329", "F410", "F329", "F410"]
labels = [0, 0, 1, 1, 1, 1]  # 1 = suicidal ideation

# Group codes
grouped = [group_diagnosis(code) for code in diagnosis_codes]

# Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(grouped)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
