import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load the dataset
data = pd.read_csv('C:\Users\NAMO\Desktop\datascience mini project\bank_transactions')

# Exploratory Data Analysis (EDA)
# Display the first few rows of the dataset
print(data.head())

# Basic statistics summary of numerical features
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Visualize the distribution of transaction amounts
plt.figure(figsize=(10, 6))
sns.histplot(data['Amount'], bins=30, kde=False, color='darkblue')
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount')
plt.ylabel('Count')
plt.show()

# Visualize the distribution of fraudulent vs non-fraudulent transactions
plt.figure(figsize=(6, 6))
sns.countplot(x='Fraudulent', data=data, palette='RdBu_r')
plt.title('Count of Fraudulent vs Non-Fraudulent Transactions')
plt.xlabel('Fraudulent')
plt.ylabel('Count')
plt.show()

# Data Preprocessing
# Drop irrelevant features
data = data.drop(['TransactionID', 'CustomerID', 'TransactionDate'], axis=1)

# Handle missing values
data = data.dropna()

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['MerchantCategory'])

# Split the data into features (X) and target (y)
X = data.drop('Fraudulent', axis=1)
y = data['Fraudulent']

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Model Evaluation
# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Cross-validation
# Perform cross-validation
cv_scores = cross_val_score(rf_classifier, X_scaled, y, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))
#This code covers the entire project workflow, including data loading, exploratory data analysis (EDA), data preprocessing, model training, evaluation, and cross-validation. It uses a Random Forest Classifier for fraud detection, preprocesses the data by handling missing values, one-hot encoding categorical variables, and normalizing numerical features. Finally, it evaluates the model's performance using accuracy, classification report, confusion matrix, and cross-validation.





