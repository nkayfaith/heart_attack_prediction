# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:28:42 2022

@author: nkayf
"""
#%% Imports and Paths

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os

DATA_PATH = (os.path.join(os.path.dirname(__file__), '..','data', 'heart.csv'))
MMS_SCALER_PATH = os.path.join(os.getcwd(),'mms_scaler.pkl')
MODEL_PATH = os.path.join(os.getcwd(),'best_model.pkl')

#%% Step 1) Data Loading

df = pd.read_csv(DATA_PATH)
column_names = list(df.columns)
df.columns = column_names

#%% Step 2) Data Interpretation/Inspection

df.info()
df.describe().T
df.isnull().sum()
df.isna().sum() 
df.boxplot() 
df[df.duplicated()]

# =============================================================================
# - check missing value : No null
# - check datatype : No NaN, uniform datatypes by columns
# - check outliers : identified for columns trtbps,chol,thalachh
# - check duplicate: identified one(1) row
# =============================================================================

#%% Step 3) Data Cleaning

df = df.drop_duplicates()
# =============================================================================
# - remove duplicate (does not affect more than 5% of dataset)
# =============================================================================

#%% Step 4) Feature Selection

X = df.drop(labels=["output"], axis=1) 
y = df["output"]
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature importance using RandomForestClassifier
forest = RandomForestClassifier()
forest.fit(X, y)

importances = forest.feature_importances_

sorted_indices = np.argsort(importances)[::-1]
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()

embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=X.shape[1])
embeded_rf_selector.fit(X, y)
embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()

# =============================================================================
# No features are selected since the highest feature importance scores less than 15%
# =============================================================================

#%% Step 5) Data Preprocessing

X = df.drop(labels=["output"], axis=1)
y = df["output"]

# Encode
le = LabelEncoder()
y = le.fit_transform(np.expand_dims(y,axis=-1)) 

# Scale
mms_scaler = MinMaxScaler()
X = mms_scaler.fit_transform(X)
pickle.dump(mms_scaler, open(MMS_SCALER_PATH,'wb'))
y = mms_scaler.fit_transform(np.expand_dims(y,axis=-1))

# =============================================================================
# Encode label
# Scale all features using minmax because data contains no negative values
# =============================================================================

#%% Step 6) Data Training

model = RandomForestClassifier(max_depth=7)

# Train the model using the training sets
model.fit(X_train,y_train)

#Predict Output
y_pred = model.predict(X_test)
print(y_pred)

# =============================================================================
# RandomForestClassifier model is selected because it scores highest in accuracy score (refer model_selection_from_score.png)
# =============================================================================

#%% Step 7) Data Evaluation

print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)

# =============================================================================
# Achieve accuracy of 86.88%
# =============================================================================

#%% Save the best approach
pickle.dump(model, open(MODEL_PATH, 'wb'))