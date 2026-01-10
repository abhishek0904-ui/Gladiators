#------------------IMPORT NECESSARY LIBRARIES--------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings as wr
wr.filterwarnings(action="ignore")
import seaborn as sns

#-------------------LOAD THE DATASETS--------------------------------
account = pd.read_csv("/workspaces/Gladiators/.venv/account_activity.csv")
customer = pd.read_csv("/workspaces/Gladiators/.venv/customer_data.csv")
fraud = pd.read_csv("/workspaces/Gladiators/.venv/fraud_indicators.csv")
suspision = pd.read_csv("/workspaces/Gladiators/.venv/suspicious_activity.csv")
merchant = pd.read_csv("/workspaces/Gladiators/.venv/merchant_data.csv")
tran_cat = pd.read_csv("/workspaces/Gladiators/.venv/transaction_category_labels.csv")
amount = pd.read_csv("/workspaces/Gladiators/.venv/amount_data.csv")
anamoly = pd.read_csv("/workspaces/Gladiators/.venv/anomaly_scores.csv")
tran_data = pd.read_csv("/workspaces/Gladiators/.venv/transaction_metadata.csv")
tran_rec = pd.read_csv("/workspaces/Gladiators/.venv/transaction_records.csv")

data = [account,customer,fraud,suspision,merchant,tran_cat,amount,anamoly,tran_data,tran_rec]
for df in data:
    print(df.head())

#--------------- Merging Customer Data-------------
costumer_data = pd.merge(customer, account, on='CustomerID')
costumer_data = pd.merge(costumer_data, suspision, on='CustomerID')
print(costumer_data)

#----------------Merging Transaction Data------------
transaction_data1 = pd.merge(fraud, tran_cat, on="TransactionID")
transaction_data2 = pd.merge(amount, anamoly, on="TransactionID")
transaction_data3 = pd.merge(tran_data, tran_rec, on="TransactionID")
transaction_data = pd.merge(transaction_data1, transaction_data2,on="TransactionID")
transaction_data = pd.merge(transaction_data, transaction_data3,on="TransactionID")
print(transaction_data)

data = pd.merge(transaction_data, costumer_data,on="CustomerID")
print(data)

#-------------------EXPLORING THE DATA-------------------------

print(data.info())
print(data.shape)
print(data.describe())

print(data.columns)

numerical_features = data.select_dtypes(include=['number']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
print(numerical_features)
print(categorical_features)

#----------------DATA VISUALIZATION----------------------
for column in data.columns:
    if data[column].dtype == 'object':  # Check if the column has a categorical data type
        top_10_values = data[column].value_counts().head(10)  # Get the first 10 unique values and their counts
        plt.figure(figsize=(10, 5))  # Adjust the figure size if needed
        sns.countplot(x=column, data=data, order=top_10_values.index)
        plt.title(f'Count Plot for {column}')
        plt.xticks(rotation=90)  # Rotate x-axis labels if they are long
        plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Get the number of numerical columns
num_cols = len(data.select_dtypes(include=['number']).columns)

num_rows = (num_cols // 2) + (num_cols % 2)

# Create subplots
fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6*num_rows))
fig.suptitle("Box Plots for Numerical Columns")

# Loop through the numerical columns and create box plots
for i, column in enumerate(data.select_dtypes(include=['number']).columns):
    row = i // 2
    col = i % 2
    sns.boxplot(x=data[column], ax=axes[row, col])
    axes[row, col].set_title(column)
# Remove any empty subplots
if num_cols % 2 != 0:
    fig.delaxes(axes[num_rows-1, 1])

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()

# We should use countplot for SuspiciousFlag feature

plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
sns.countplot(x='SuspiciousFlag', data=data, palette='Set2')  # You can change the palette as desired
plt.title('Count Plot for Suspicious Flag')
plt.xlabel('Suspicious Flag')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels if they are long

plt.show()

#As we can see the dataset's target feature is heavily imbalanced so we can use further techiniqes to equalize the feature's values

# Select only the numeric columns
numeric_data = data.select_dtypes(include=['number'])

# Calculate the correlation matrix for numeric columns
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap for Numeric Columns')

plt.show()


#---------------Benford's Law Application-----------------

# Extract leading digit and map to Benford Probability
def get_first_digit(amount):
    s = str(abs(amount)).replace('.', '').lstrip('0')
    return int(s[0]) if s else 0

data['First_Digit'] = data['TransactionAmount'].apply(get_first_digit)
benford_dist = {d: np.log10(1 + 1/d) for d in range(1, 10)}
data['Benford_Prob'] = data['First_Digit'].map(benford_dist).fillna(0)
columns_to_be_dropped = ['TransactionID','MerchantID','CustomerID','Name', 'Age', 'Address']

data1 = data.drop(columns_to_be_dropped, axis=1)
print(data1.head())

print(data1['FraudIndicator'].value_counts(), data1['SuspiciousFlag'].value_counts(), data1['Category'].value_counts())

#--------------------------FEATURE ENGINEERING-------------------------

# 1. Convert columns to datetime in 'data1' directly
data1['Timestamp'] = pd.to_datetime(data1['Timestamp'])
data1['LastLogin'] = pd.to_datetime(data1['LastLogin'])

# 2. Create Time-based features on 'data1'
data1['Hour'] = data1['Timestamp'].dt.hour
data1['gap'] = (data1['Timestamp'] - data1['LastLogin']).dt.days.abs()

# 3. Create Amount-based features on 'data1'
# Returns 1 if it's a suspicious "clean" number, 0 otherwise
data1['Is_Round_Amount'] = data1['TransactionAmount'].apply(lambda x: 1 if x % 100 == 0 or x % 500 == 0 else 0)

# 4. Day of Week features
# 0=Monday, 6=Sunday
data1['DayOfWeek'] = data1['Timestamp'].dt.dayofweek

# Flag transactions happening on weekends (Saturday=5, Sunday=6)
data1['Is_Weekend'] = data1['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Flag "Graveyard Shift" (Transactions between 1 AM and 5 AM)
data1['Is_Night_Trans'] = data1['Hour'].apply(lambda x: 1 if 1 <= x <= 5 else 0)

# 5. Advanced Profile Features
.
columns_to_drop_final = ['TransactionID','MerchantID','CustomerID','Name', 'Age', 'Address', 'Timestamp', 'LastLogin']

data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['LastLogin'] = pd.to_datetime(data['LastLogin'])
data['Hour'] = data['Timestamp'].dt.hour
data['gap'] = (data['Timestamp'] - data['LastLogin']).dt.days.abs()
data['Is_Round_Amount'] = data['TransactionAmount'].apply(lambda x: 1 if x % 100 == 0 or x % 500 == 0 else 0)
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
data['Is_Weekend'] = data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
data['Is_Night_Trans'] = data['Hour'].apply(lambda x: 1 if 1 <= x <= 5 else 0)
data['Cust_Avg_Amount'] = data.groupby('CustomerID')['TransactionAmount'].transform('mean')
data['High_Value_Spike'] = (data['TransactionAmount'] > 5 * data['Cust_Avg_Amount']).astype(int)
data['Weekend_Spike'] = data['High_Value_Spike'] * data['Is_Weekend']

data1 = data.drop(columns_to_drop_final, axis=1)

print(data1.head())

#------------------DATA MODELLING----------------------------------

# Separate Target and Features
X = data1.drop(['FraudIndicator'], axis=1) # Timestamp/LastLogin were already dropped above
Y = data1['FraudIndicator']

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

X['Category'] = X['Category'].astype(str)
X['Category'] = label_encoder.fit_transform(X['Category'])

# Handle NaNs if any were created 
X = X.fillna(0)

print("Features in X:", X.columns) 

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(X_train.shape, Y_test.shape)


# Logistic Regression model

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# # Create a logistic regression model
# model = DecisionTreeClassifier()

# # Train the model on the training data
# model.fit(X_train, Y_train)

# # Make predictions on the testing data
# y_pred = model.predict(X_test)

# # Calculate the accuracy of the model
# accuracy = accuracy_score(Y_test, y_pred)
# print("Accuracy:", accuracy)

# #High accuracy is not often a good thing in a machine learning model as it states the problem of imbalanced dataset

import seaborn as sns
import matplotlib.pyplot as plt


# Create a count plot for the 'FraudIndicator' column
plt.figure(figsize=(8, 6))
sns.countplot(data=data1, x='FraudIndicator', palette='viridis')
plt.title('Count Plot of Fraud Indicator')
plt.xlabel('Fraud Indicator')
plt.ylabel('Count')
print(plt.show())


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter


# Initialize SMOTE for oversampling
smote = SMOTE(random_state=42)

# Apply SMOTE to the data
X_resampled, y_resampled = smote.fit_resample(X, Y)

print("Class distribution after oversampling:", Counter(y_resampled))

plt.figure(figsize=(8, 6))
sns.countplot(data=pd.DataFrame({'FraudIndicator': y_resampled}), x='FraudIndicator', palette='viridis')
plt.title('Count Plot of Fraud Indicator (After Oversampling)')
plt.xlabel('Fraud Indicator')
plt.ylabel('Count')
plt.show()

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import shuffle

X_resampled, y_resampled = shuffle(X_resampled, y_resampled, random_state=42)

# Split data into train and test sets (keep the test set separate for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Initialize the Random Forest Classifier
# We use 100 trees (n_estimators) and a fixed random_state for reproducibility
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. Train the model on the resampled (balanced) training data
rf_model.fit(X_train, y_train)

# 3. Make predictions on the test data
y_pred_rf = rf_model.predict(X_test)

# 4. Evaluate the model performance
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# precision_rf = precision_score(y_test, y_pred_rf)
# recall_rf = recall_score(y_test, y_pred_rf)
# f1_rf = f1_score(y_test, y_pred_rf)
# conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# print("Random Forest Model Evaluation Metrics:")
# print("Accuracy:",accuracy_rf)
# print("Precision:", precision_rf)
# print("Recall:",recall_rf)
# print("F1 Score:", f1_rf)
# print("Confusion Matrix:")
# print(conf_matrix_rf)

# # Detailed Classification Report
# print("Detailed Classification Report:")
# print(classification_report(y_test, y_pred_rf))

#----------- Cross-Validation to assess model stability----------------

from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

# 1. Define the Cross-Validation strategy
# We use StratifiedKFold to ensure each fold has a balanced percentage of fraud
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Calculate Cross-Validation scores for 'Recall' 
recall_scores = cross_val_score(rf_model, X_resampled, y_resampled, cv=skf, scoring='recall')

# 3. Calculate Cross-Validation scores for 'Accuracy'
accuracy_scores = cross_val_score(rf_model, X_resampled, y_resampled, cv=skf, scoring='accuracy')

print("--- Cross-Validation Results (5-Folds) ---")
print(f"Average Recall: {np.mean(recall_scores):.4f} (+/- {np.std(recall_scores):.4f})")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")

import joblib


# 'rf_model' is the variable name from your Random Forest training code
joblib.dump(rf_model, 'fraud_model.pkl')

# The backend needs this to translate new data exactly the same way.
joblib.dump(label_encoder, 'category_encoder.pkl')

# This saves the exact list of columns your model expects (e.g., 'Hour', 'gap', 'Amount')
model_columns = X.columns.tolist()
joblib.dump(model_columns, 'model_features.pkl')

print("Success! Files 'fraud_model.pkl', 'category_encoder.pkl', and 'model_features.pkl' are ready for the backend.")
# Inference on new/unseen data (for example, use a separate unseen dataset or a specific test sample)
# Here we simulate it by using the first row from the X_test
unseen_sample = X_test.iloc[69].values.reshape(1, -1)  # Reshaping for a single sample

# Predict the label for the unseen sample
inference_prediction = rf_model.predict(unseen_sample)

# Map prediction result to 'fraud' or 'not fraud'
fraud_status = "Fraud" if inference_prediction[0] == 1 else "Not Fraud"

print("Inference Prediction for Unseen Sample:", fraud_status)


#------------------IMPORT NECESSARY LIBRARIES--------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings as wr
wr.filterwarnings(action="ignore")
import seaborn as sns

#-------------------LOAD THE DATASETS--------------------------------
account = pd.read_csv("/workspaces/Gladiators/.venv/account_activity.csv")
customer = pd.read_csv("/workspaces/Gladiators/.venv/customer_data.csv")
fraud = pd.read_csv("/workspaces/Gladiators/.venv/fraud_indicators.csv")
suspision = pd.read_csv("/workspaces/Gladiators/.venv/suspicious_activity.csv")
merchant = pd.read_csv("/workspaces/Gladiators/.venv/merchant_data.csv")
tran_cat = pd.read_csv("/workspaces/Gladiators/.venv/transaction_category_labels.csv")
amount = pd.read_csv("/workspaces/Gladiators/.venv/amount_data.csv")
anamoly = pd.read_csv("/workspaces/Gladiators/.venv/anomaly_scores.csv")
tran_data = pd.read_csv("/workspaces/Gladiators/.venv/transaction_metadata.csv")
tran_rec = pd.read_csv("/workspaces/Gladiators/.venv/transaction_records.csv")

data = [account,customer,fraud,suspision,merchant,tran_cat,amount,anamoly,tran_data,tran_rec]
for df in data:
    print(df.head())

#--------------- Merging Customer Data-------------
costumer_data = pd.merge(customer, account, on='CustomerID')
costumer_data = pd.merge(costumer_data, suspision, on='CustomerID')
print(costumer_data)

#----------------Merging Transaction Data------------
transaction_data1 = pd.merge(fraud, tran_cat, on="TransactionID")
transaction_data2 = pd.merge(amount, anamoly, on="TransactionID")
transaction_data3 = pd.merge(tran_data, tran_rec, on="TransactionID")
transaction_data = pd.merge(transaction_data1, transaction_data2,on="TransactionID")
transaction_data = pd.merge(transaction_data, transaction_data3,on="TransactionID")
print(transaction_data)

data = pd.merge(transaction_data, costumer_data,on="CustomerID")
print(data)

#-------------------EXPLORING THE DATA-------------------------

print(data.info())
print(data.shape)
print(data.describe())

print(data.columns)

numerical_features = data.select_dtypes(include=['number']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
print(numerical_features)
print(categorical_features)

#----------------DATA VISUALIZATION----------------------
for column in data.columns:
    if data[column].dtype == 'object':  # Check if the column has a categorical data type
        top_10_values = data[column].value_counts().head(10)  # Get the first 10 unique values and their counts
        plt.figure(figsize=(10, 5))  # Adjust the figure size if needed
        sns.countplot(x=column, data=data, order=top_10_values.index)
        plt.title(f'Count Plot for {column}')
        plt.xticks(rotation=90)  # Rotate x-axis labels if they are long
        plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Get the number of numerical columns
num_cols = len(data.select_dtypes(include=['number']).columns)

# Calculate the number of rows and columns for subplots
num_rows = (num_cols // 2) + (num_cols % 2)

# Create subplots
fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6*num_rows))
fig.suptitle("Box Plots for Numerical Columns")

# Loop through the numerical columns and create box plots
for i, column in enumerate(data.select_dtypes(include=['number']).columns):
    row = i // 2
    col = i % 2
    sns.boxplot(x=data[column], ax=axes[row, col])
    axes[row, col].set_title(column)
# Remove any empty subplots
if num_cols % 2 != 0:
    fig.delaxes(axes[num_rows-1, 1])

plt.tight_layout()
plt.subplots_adjust(top=0.95)  # Adjust the position of the overall title
plt.show()

# We should use countplot for SuspiciousFlag feature

plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
sns.countplot(x='SuspiciousFlag', data=data, palette='Set2')  # You can change the palette as desired
plt.title('Count Plot for Suspicious Flag')
plt.xlabel('Suspicious Flag')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels if they are long

plt.show()

#As we can see the dataset's target feature is heavily imbalanced so we can use further techiniqes to equalize the feature's values

# Select only the numeric columns
numeric_data = data.select_dtypes(include=['number'])

# Calculate the correlation matrix for numeric columns
correlation_matrix = numeric_data.corr()

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap for Numeric Columns')

plt.show()

#---------------Benford's Law Application-----------------

data['First_Digit'] = data['TransactionAmount'].apply(get_first_digit)
benford_dist = {d: np.log10(1 + 1/d) for d in range(1, 10)}
data['Benford_Prob'] = data['First_Digit'].map(benford_dist).fillna(0)
columns_to_be_dropped = ['TransactionID','MerchantID','CustomerID','Name', 'Age', 'Address']

data1 = data.drop(columns_to_be_dropped, axis=1)
print(data1.head())

print(data1['FraudIndicator'].value_counts(), data1['SuspiciousFlag'].value_counts(), data1['Category'].value_counts())

#--------------------------FEATURE ENGINEERING-------------------------

# 1. Convert columns to datetime in 'data1' directly
data1['Timestamp'] = pd.to_datetime(data1['Timestamp'])
data1['LastLogin'] = pd.to_datetime(data1['LastLogin'])

# 2. Create Time-based features on 'data1'
data1['Hour'] = data1['Timestamp'].dt.hour
data1['gap'] = (data1['Timestamp'] - data1['LastLogin']).dt.days.abs()

# 3. Create Amount-based features on 'data1'
data1['Is_Round_Amount'] = data1['TransactionAmount'].apply(lambda x: 1 if x % 100 == 0 or x % 500 == 0 else 0)

# 4. Day of Week features
# 0=Monday, 6=Sunday
data1['DayOfWeek'] = data1['Timestamp'].dt.dayofweek

# Flag transactions happening on weekends (Saturday=5, Sunday=6)
data1['Is_Weekend'] = data1['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Flag "Graveyard Shift" (Transactions between 1 AM and 5 AM)
data1['Is_Night_Trans'] = data1['Hour'].apply(lambda x: 1 if 1 <= x <= 5 else 0)

# 5. Advanced Profile Features
columns_to_drop_final = ['TransactionID','MerchantID','CustomerID','Name', 'Age', 'Address', 'Timestamp', 'LastLogin']

data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['LastLogin'] = pd.to_datetime(data['LastLogin'])
data['Hour'] = data['Timestamp'].dt.hour
data['gap'] = (data['Timestamp'] - data['LastLogin']).dt.days.abs()
data['Is_Round_Amount'] = data['TransactionAmount'].apply(lambda x: 1 if x % 100 == 0 or x % 500 == 0 else 0)
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
data['Is_Weekend'] = data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
data['Is_Night_Trans'] = data['Hour'].apply(lambda x: 1 if 1 <= x <= 5 else 0)
data['Cust_Avg_Amount'] = data.groupby('CustomerID')['TransactionAmount'].transform('mean')
data['High_Value_Spike'] = (data['TransactionAmount'] > 5 * data['Cust_Avg_Amount']).astype(int)
data['Weekend_Spike'] = data['High_Value_Spike'] * data['Is_Weekend']

# We drop 'Timestamp' and 'LastLogin' here because we already extracted features from them
data1 = data.drop(columns_to_drop_final, axis=1)

print(data1.head())

#------------------DATA MODELLING----------------------------------

# Separate Target and Features
X = data1.drop(['FraudIndicator'], axis=1) # Timestamp/LastLogin were already dropped above
Y = data1['FraudIndicator']

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

X['Category'] = X['Category'].astype(str)
X['Category'] = label_encoder.fit_transform(X['Category'])

# Handle NaNs if any were created (e.g., by shift or map)
X = X.fillna(0)

print("Features in X:", X.columns) 

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(X_train.shape, Y_test.shape)


# Logistic Regression model

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# # Create a logistic regression model
# model = DecisionTreeClassifier()

# # Train the model on the training data
# model.fit(X_train, Y_train)

# # Make predictions on the testing data
# y_pred = model.predict(X_test)

# # Calculate the accuracy of the model
# accuracy = accuracy_score(Y_test, y_pred)
# print("Accuracy:", accuracy)

# #High accuracy is not often a good thing in a machine learning model as it states the problem of imbalanced dataset

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame 'df' with a 'FraudIndicator' column
# Load your data into the DataFrame if not already done

# Create a count plot for the 'FraudIndicator' column
plt.figure(figsize=(8, 6))  # Optional: Adjust the figure size
sns.countplot(data=data1, x='FraudIndicator', palette='viridis')
plt.title('Count Plot of Fraud Indicator')
plt.xlabel('Fraud Indicator')
plt.ylabel('Count')
print(plt.show())


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter


# Initialize SMOTE for oversampling
smote = SMOTE(random_state=42)

# Apply SMOTE to the data
X_resampled, y_resampled = smote.fit_resample(X, Y)

# Check the class distribution after oversampling
print("Class distribution after oversampling:", Counter(y_resampled))

# Create a count plot for the 'FraudIndicator' column after oversampling
plt.figure(figsize=(8, 6))
sns.countplot(data=pd.DataFrame({'FraudIndicator': y_resampled}), x='FraudIndicator', palette='viridis')
plt.title('Count Plot of Fraud Indicator (After Oversampling)')
plt.xlabel('Fraud Indicator')
plt.ylabel('Count')
plt.show()

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import shuffle

# Shuffle the dataset to introduce slight randomness (you can adjust the `random_state` for different outcomes)
X_resampled, y_resampled = shuffle(X_resampled, y_resampled, random_state=42)

# Split data into train and test sets (keep the test set separate for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Initialize the Random Forest Classifier
# We use 100 trees (n_estimators) and a fixed random_state for reproducibility
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. Train the model on the resampled (balanced) training data
rf_model.fit(X_train, y_train)

# 3. Make predictions on the test data
y_pred_rf = rf_model.predict(X_test)

# 4. Evaluate the model performance
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# precision_rf = precision_score(y_test, y_pred_rf)
# recall_rf = recall_score(y_test, y_pred_rf)
# f1_rf = f1_score(y_test, y_pred_rf)
# conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# print("Random Forest Model Evaluation Metrics:")
# print("Accuracy:",accuracy_rf)
# print("Precision:", precision_rf)
# print("Recall:",recall_rf)
# print("F1 Score:", f1_rf)
# print("Confusion Matrix:")
# print(conf_matrix_rf)

# # Detailed Classification Report
# print("Detailed Classification Report:")
# print(classification_report(y_test, y_pred_rf))

#----------- Cross-Validation to assess model stability----------------

from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

# 1. Define the Cross-Validation strategy
# We use StratifiedKFold to ensure each fold has a balanced percentage of fraud
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Calculate Cross-Validation scores for 'Recall' 
recall_scores = cross_val_score(rf_model, X_resampled, y_resampled, cv=skf, scoring='recall')

# 3. Calculate Cross-Validation scores for 'Accuracy'
accuracy_scores = cross_val_score(rf_model, X_resampled, y_resampled, cv=skf, scoring='accuracy')

print("--- Cross-Validation Results (5-Folds) ---")
print(f"Average Recall: {np.mean(recall_scores):.4f} (+/- {np.std(recall_scores):.4f})")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")

import joblib


# 'rf_model' is the variable name from your Random Forest training code
joblib.dump(rf_model, 'fraud_model.pkl')

# The backend needs this to translate new data exactly the same way.
joblib.dump(label_encoder, 'category_encoder.pkl')

# This saves the exact list of columns your model expects (e.g., 'Hour', 'gap', 'Amount')
model_columns = X.columns.tolist()
joblib.dump(model_columns, 'model_features.pkl')

print("Success! Files 'fraud_model.pkl', 'category_encoder.pkl', and 'model_features.pkl' are ready for the backend.")
# Inference on new/unseen data (for example, use a separate unseen dataset or a specific test sample)
# Here we simulate it by using the first row from the X_test
unseen_sample = X_test.iloc[69].values.reshape(1, -1)  # Reshaping for a single sample

# Predict the label for the unseen sample
inference_prediction = rf_model.predict(unseen_sample)

# Map prediction result to 'fraud' or 'not fraud'
fraud_status = "Fraud" if inference_prediction[0] == 1 else "Not Fraud"

print("Inference Prediction for Unseen Sample:", fraud_status)


