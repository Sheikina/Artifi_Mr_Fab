import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE


data = pd.read_excel("C:/Users/$he!ikscrown/Documents/AI with Mr Fabrice/dataset.xlsx")

# Check unique values in churn column
print("Unique values in churn column:", np.unique(data["churn"]))

# 2. Data Exploration and Preprocessing
def preprocess_data(data):
    # Handle missing values
    data.ffill(inplace=True)  # Forward fill to handle missing values

    # Encode categorical variables
    label_encoders = {}
    for column in data.select_dtypes(include=["object"]).columns:
        data[column] = data[column].astype(str)
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Scale numerical features
    scaler = StandardScaler()
    num_columns = data.select_dtypes(include=["float64", "int64"]).columns
    data[num_columns] = scaler.fit_transform(data[num_columns])

    return data, label_encoders, scaler

# Preprocess the data
data, label_encoders, scaler = preprocess_data(data)

# Split into train and test sets (70% train, 30% test)
X = data.drop(columns=["churn"])  # Assuming 'churn' is the target variable
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ensure y_train and y_test contain only integers
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print("Unique values in y_train:", np.unique(y_train))
print("Unique values in y_test:", np.unique(y_test))

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Model Training
initial_model = LogisticRegression(max_iter=1000)
initial_model.fit(X_train, y_train)

# Get predicted probabilities
y_pred_proba_initial = initial_model.predict_proba(X_test)[:, 1]

# Convert probabilities to binary predictions (0 or 1) using a threshold of 0.5
y_pred_initial = (y_pred_proba_initial >= 0.5).astype(int)

# Check unique values in y_pred_initial
print("Unique values in y_pred_initial:", np.unique(y_pred_initial))

print("Initial Model Performance on Test Data")
print(classification_report(y_test, y_pred_initial))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba_initial))  # Use probabilities for AUC-ROC

# Concept Drift and Data Shifts
weights = np.linspace(1.0, 2.0, len(y_train))
model_time_weighted = GradientBoostingClassifier()
model_time_weighted.fit(X_train, y_train, sample_weight=weights)

model_ensemble = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=1000)),
    ('dt', DecisionTreeClassifier()),
    ('gb', GradientBoostingClassifier())
], voting='soft')
model_ensemble.fit(X_train, y_train)

model_sgd = SGDClassifier()
model_sgd.partial_fit(X_train, y_train, classes=np.unique(y_train))

# Evaluate Model Adaptation
y_pred_time_weighted = model_time_weighted.predict(X_test)
y_pred_ensemble = model_ensemble.predict(X_test)
y_pred_sgd = model_sgd.predict(X_test)

print("Time-Weighted Model Performance")
print(classification_report(y_test, y_pred_time_weighted))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_time_weighted))

print("Ensemble Model Performance")
print(classification_report(y_test, y_pred_ensemble))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_ensemble))

print("Online Model Performance (SGD)")
print(classification_report(y_test, y_pred_sgd))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_sgd))

# Monitoring and Performance Metrics
performance_metrics = {
    "accuracy": accuracy_score(y_test, y_pred_ensemble),
    "precision": precision_recall_fscore_support(y_test, y_pred_ensemble, average="binary")[0],
    "recall": precision_recall_fscore_support(y_test, y_pred_ensemble, average="binary")[1],
    "f1": precision_recall_fscore_support(y_test, y_pred_ensemble, average="binary")[2],
    "auc_roc": roc_auc_score(y_test, y_pred_ensemble)
}

print(performance_metrics)
