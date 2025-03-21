# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('customer_churn.csv')

# Display the first few rows of the dataset
print("Dataset Sample:\n", df.head())

# Basic data preprocessing: handle missing values (if any)
df = df.dropna()

# Convert categorical columns to numeric using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['gender', 'contract_type', 'payment_method'])

# Define features (X) and target (y)
X = df_encoded.drop('churn', axis=1)  # Features
y = df_encoded['churn']               # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling for numerical stability
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Logistic Regression classifier
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=1)

print(f"\nAccuracy: {accuracy}")
print("\nClassification Report:\n", report)
