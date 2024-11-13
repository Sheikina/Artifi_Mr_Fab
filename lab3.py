import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


pd.set_option('display.max_columns', None)


df = pd.read_csv("C:/Users/$he!ikscrown/Documents/AI with Mr Fabrice/titanic.csv")


print("Initial DataFrame with missing values:\n", df.isnull().sum())

# Fill missing values
for column in df.columns:
    if df[column].dtype == 'float64' or df[column].dtype == 'int64':
        # Use mean for numerical columns
        df[column].fillna(df[column].mean(), inplace=True)
    elif df[column].dtype == 'object':
        # Use mode for categorical columns
        df[column].fillna(df[column].mode()[0], inplace=True)

#  no more missing values
print("\nDataFrame after filling missing values:\n", df.isnull().sum())


numerical_features = ['age', 'fare']
min_max_scaler = MinMaxScaler()
df[numerical_features] = min_max_scaler.fit_transform(df[numerical_features])
print("\nDataFrame after Min-Max Scaling on 'age' and 'fare':\n", df[['age', 'fare']].head())

# Create 'family_size'
df['family_size'] = df['sibsp'] + df['parch']
print("\nFinal Updated DataFrame with 'family_size' column:\n", df[['sibsp', 'parch', 'family_size']].head())

# Label Encoding for categorical columns
label_encoder = LabelEncoder()


df['sex'] = label_encoder.fit_transform(df['sex'])
df['embarked'] = label_encoder.fit_transform(df['embarked'].astype(str))  # Ensure 'embarked' is a string type

# Remove non-numeric columns before correlation (such as 'name', 'ticket', 'cabin', etc.)
df_encoded = pd.get_dummies(df, drop_first=True)

# Define features and target variable
X = df_encoded.drop(columns=['survived'])
y = df_encoded['survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


lr_model = LogisticRegression(max_iter=500, random_state=42)
lr_model.fit(X_train, y_train)

# Predict on the test set
y_pred = lr_model.predict(X_test)

# Evaluate Logistic Regression Model
lr_accuracy = accuracy_score(y_test, y_pred)
print("\nLogistic Regression Model Accuracy: {:.2f}%".format(lr_accuracy * 100))

# Confusion Matrix for Logistic Regression
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Not Survived', 'Survived'])
disp.plot(cmap='Blues')
plt.title("Logistic Regression - Confusion Matrix")
plt.show()
