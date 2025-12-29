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

data = [account, customer, fraud, suspision, merchant, tran_cat, amount, anamoly, tran_data, tran_rec]
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


