# =============================================================================
# model.py
# Medical Appointment No-Show Prediction — SVM Model
# Author: Athira Sivadas | Entri Elevate
# =============================================================================

# import necessary libraries

import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA LOADING
# =============================================================================
# Load the dataset
df = pd.read_csv("KaggleV2-May-2016.csv")
# Display first 5 rows
print(df.head())

# =============================================================================
# 2. PREPROCESSING & FEATURE ENGINEERING
# =============================================================================

# Convert Date Columns

df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.normalize()
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.normalize()

#  Create Useful Feature (Waiting Days)
df['WaitingDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

# --- Remove data-entry errors (negative waiting days) ---
before = df.shape[0]
df = df[df['WaitingDays'] >= 0].copy()
print(f"Removed {before - df.shape[0]} rows with negative WaitingDays")

# --- Reduce skewness with log transform ---
df['WaitingDays'] = np.log1p(df['WaitingDays'])

# --- Encode Gender (F=0, M=1) ---
# Encode Gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Save encoder
with open('gender_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# --- Encode target variable (No=0, Yes=1) ---
noshow_map = {'No': 0, 'Yes': 1}
df['No-show'] = df['No-show'].map(noshow_map)
print(f"Preprocessing complete. Final dataset: {df.shape[0]} rows")

# =============================================================================
# 3.  FEATURE SCALING
# =============================================================================

#StandardScaler
#StandardScaler transforms the data so that the mean becomes 0 and the standard deviation becomes 1.
scaler = StandardScaler()

df[['Age','WaitingDays']] = scaler.fit_transform(df[['Age','WaitingDays']])

#save the scalar
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


# =============================================================================
# 4. TRAIN SVM MODEL
# =============================================================================
# Splitting the Dataset

# Features and target variable
X = df.drop(['No-show', 'PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay', 'Neighbourhood'], axis=1)
Y = df['No-show']

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Training the model

svm_model = SVC()
svm_model.fit(X_train, Y_train)

# save the model 
pickle.dump(svm_model,open('model.pkl','wb'))
