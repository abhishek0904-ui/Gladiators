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
# Assuming 'data' is your DataFrame containing numerical columns

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

# Dropping the columns as of now they are not mush corelated & also wouldn't damper the performance of model

columns_to_be_dropped = ['TransactionID','MerchantID','CustomerID','Name', 'Age', 'Address']

data1 = data.drop(columns_to_be_dropped, axis=1)
print(data1.head())

print(data1['FraudIndicator'].value_counts(), data1['SuspiciousFlag'].value_counts(), data1['Category'].value_counts())

#--------------------------FEATURE ENGINEERING-------------------------

# Using Feature Engineering Creating two Columns
# Hour of Transaction = hour
# Gap between the day of transaction and last login in days = gap
if pd.api.types.is_datetime64_any_dtype(data['Timestamp']):
    print("The 'Timestamp' column is already in datetime format.")
else:
    print("The 'Timestamp' column is not in datetime format.")

data1['Timestamp1'] = pd.to_datetime(data1['Timestamp'])

print(data1.dtypes)
data1['Hour'] = data1['Timestamp1'].dt.hour
data1['LastLogin'] = pd.to_datetime(data1['LastLogin'])
data1['gap'] = (data1['Timestamp1'] - data1['LastLogin']).dt.days.abs()

print(data1.head())

#------------------DATA MODELLING----------------------------------
X = data1.drop(['FraudIndicator','Timestamp','Timestamp1','LastLogin'],axis=1)
Y = data1['FraudIndicator']


from sklearn.preprocessing import LabelEncoder
# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'Category' column
X['Category'] = label_encoder.fit_transform(X['Category'])
print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

print(X_train.shape, Y_test.shape)

# Logistic Regression model

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Create a logistic regression model
model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_train, Y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

#High accuracy is not often a good thing in a machine learning model as it states the problem of imbalanced dataset

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

#--------------------SMOTE ANALYSIS--------------------------------------
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


# Retraining Logistic regression using SAMPLED Data

model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_resampled, y_resampled)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate and print various metrics to evaluate the model's performance
accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)
confusion = confusion_matrix(Y_test, y_pred)

print("Model Evaluation Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion)


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import shuffle

# Shuffle the dataset to introduce slight randomness (you can adjust the `random_state` for different outcomes)
X_resampled, y_resampled = shuffle(X_resampled, y_resampled, random_state=42)

# Split data into train and test sets (keep the test set separate for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Initialize the model
model = DecisionTreeClassifier()

# Define the hyperparameters grid for tuning
param_grid = {
    'max_depth': [None, 10, 25, 50,100,250],  # Maximum depth of the tree
    'min_samples_split': [2, 10, 50,100,250],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 10, 30,75,100],    # Minimum number of samples required to be at a leaf node
    'criterion': ['gini', 'entropy'] # Criterion to measure the quality of a split
}

# Initialize GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Perform the grid search to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Best parameters found by GridSearchCV
print("Best Hyperparameters:", grid_search.best_params_)

# Train the model using the best hyperparameters
best_model = grid_search.best_estimator_

# Make predictions on the test data
y_pred = best_model.predict(X_test)

# Calculate and print various metrics to evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("Model Evaluation Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion)


# Inference on new/unseen data (for example, use a separate unseen dataset or a specific test sample)
# Here we simulate it by using the first row from the X_test
unseen_sample = X_test.iloc[59].values.reshape(1, -1)  # Reshaping for a single sample

# Predict the label for the unseen sample
inference_prediction = best_model.predict(unseen_sample)

# Map prediction result to 'fraud' or 'not fraud'
fraud_status = "Fraud" if inference_prediction[0] == 1 else "Not Fraud"

print("Inference Prediction for Unseen Sample:", fraud_status)


