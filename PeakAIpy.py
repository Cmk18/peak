# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 00:03:55 2024

@author: dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import linregress

dataset=pd.read_csv("creditcard.csv")
print(dataset.shape)

def check_imbalance(dataset, threshold=0.33):
    df = pd.DataFrame(dataset)
    class_counts = df.iloc[:,1].value_counts()
    max_class_count = class_counts.max()
    ratio = max_class_count / len(df)
    if ratio > threshold:
        for label, count in class_counts.items():
            if count == max_class_count:
                return label
    else:
        return None

imbalanced_class = check_imbalance(dataset)
if imbalanced_class is None:
    print("The dataset is balanced")
else:
    print(f"The class '{imbalanced_class}' is imbalanced")
    
import seaborn as sns
corr = dataset.corr()
print(corr)
sns.heatmap(corr, annot = True)


cat_var = dataset.iloc[:, list(range(0, 28)) + list(range(29, 30))]
print(cat_var)

data_df = pd.DataFrame(dataset) 
num_var=dataset.iloc[:, -2]
print(num_var)

missingvalueindex=dataset.isnull().sum().sort_values(ascending=True)
print(missingvalueindex)

summary_stats = dataset.iloc[:, -2].describe()
print("Summary Statistics of the num_var:")
print(summary_stats)

# time series data analysis , and understand the data just plot the stas model 
# aggregate the data , look across all  the data whether it provides anything and exclude some of them
# data will be reperform 
# which amount is fradulent or the amount is not fradulent by creating a identifier to prevent the credit card activity

# Look at the data in a more business perspective 

# to identify the weried values base on the data

# use of some business formulation 
time_series_data = dataset[['Time', 'Amount', 'Class']].copy()

# Plotting time series data
plt.figure(figsize=(15, 5))
plt.plot(time_series_data['Time'], time_series_data['Amount'], label='Amount')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.title('Time Series Analysis of Transaction Amount')
plt.legend()
plt.show()

agg_data = dataset.groupby('Class').agg({'Amount': ['mean', 'median', 'max', 'min', 'sum', 'count']})
print(agg_data)

num_var = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
numeric_data = dataset[num_var]
outliers = {}
for col in numeric_data:
    # Calculate 1st and 3rd quartiles
    q1 = numeric_data[col].quantile(0.25)
    q3 = numeric_data[col].quantile(0.75)
    # Calculate IQR
    iqr = q3 - q1
    # Define lower and upper bounds for outliers
    lower_side = q1 - (1.5*iqr)
    upper_side = q3 +( 1.5*iqr)
    # Count number of outliers
    outliers[col] = numeric_data[(numeric_data[col] < lower_side) | (numeric_data[col] > upper_side)].shape[0]

# Print the outliers count of each variable
print(outliers)

#outlier_rows = numeric_data[(numeric_data[col] < lower_side) | (numeric_data[col] > upper_side)].index

# Drop the rows that contain outlier values
#dataset = dataset.drop(outlier_rows)
#print(dataset.shape)

class_counts = dataset['Class'].value_counts()
print("Class distribution:")
print(class_counts)

X = dataset.drop(dataset.columns[30], axis=1)
print(X)

y=dataset.iloc[:,30].values
print(y)

from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
clf=LogisticRegression(random_state=0)
clf.fit(X_train,y_train)
coef=clf.coef_
intercept=clf.intercept_
print(f'Coefficient of model:{coef}')
print(f'Intercept of model:{intercept}')

y_pred=clf.predict(X_test)
# Compare y_pred and y_test using accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Logistic Regression model:",accuracy)

#Plot a diagram to show the probability 
plt.figure()
fx = np.dot(X_test, coef.T) + intercept
p = 1/(1+np.exp(-1*fx))
plt.scatter(fx,p,color='r')
plt.xlabel('fx')
plt.ylabel('Probability')
plt.show()

compare_pred_true = pd.DataFrame({'y_True': y_test, 'y_pred': y_pred})
print(compare_pred_true)

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot heatmap
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

#Generate the clf report 
from sklearn.metrics import accuracy_score,classification_report
accuracy_score=accuracy_score(y_test,y_pred)
clf_report=classification_report(y_test,y_pred)
print(f'The accuracy score is {accuracy_score:.3f}')
print(clf_report)

# Filter dataset for fraudulent transactions (Class 1)
fraudulent_data = dataset[dataset['Class'] == 1]

# Extract input variables associated with fraudulent transactions
input_variables_fraudulent = fraudulent_data.drop('Class', axis=1)

# Find the range of input variables
min_input_variables = input_variables_fraudulent.min()
max_input_variables = input_variables_fraudulent.max()

print("Range of input variables for fraudulent transactions (Class 1):")
print("Minimum values:")
print(min_input_variables)
print("\nMaximum values:")
print(max_input_variables)



