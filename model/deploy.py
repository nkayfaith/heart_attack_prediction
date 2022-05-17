# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:29:03 2022

@author: nkayf
"""

import streamlit as st
import numpy as np
import pickle
import os

# Paths
MMS_SCALER_PATH = os.path.join(os.getcwd(),'mms_scaler.pkl')
MODEL_PATH = os.path.join(os.getcwd(),'best_model.pkl')

# Load Settings
mms_scaler = pickle.load(open(MMS_SCALER_PATH,'rb'))

# Load model
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)
    
heart_attack_chance = {0:'negative',1:'positive'}

#%% Model Evaluate

patient_info = np.array([61,1,0,140,207,0,0,138,1,1.9,2,1,3]) 
patient_info = mms_scaler.fit_transform(np.expand_dims(patient_info,axis=0))
y_pred = model.predict(patient_info)
print(y_pred)

print(heart_attack_chance[y_pred[0]])

#%% App 

with st.form('Heart Attack Prediction Form'):
    st.write('Patient\'s Info')
    age = int(st.number_input('Age'))
    sex = st.number_input('Sex')
    cp = st.number_input('Chest Pain Type')
    trtbps = st.number_input('Resting Blood Pressure')
    chol = st.number_input('Cholestoral in mg/dl fetched via BMI sensor')
    fbs = st.number_input('(Fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)')
    restecg = st.number_input('Resting Electrocardiographic Results')
    thalachh = st.number_input('Maximum Heart Rate Achieved')
    exng = st.number_input('Exercise Induced Angina (1 = yes; 0 = no)')
    oldpeak = st.number_input('Previous Peak')
    slp = st.number_input('Slope')
    caa = st.number_input('number of major vessels (0-3)')
    thall = st.number_input('Thal rate')

   
    submitted = st.form_submit_button('Submit')
    
    if submitted == True:
        #just cnp the evaluate part, mindful of DP/ML difference
        patient_info = np.array([age, sex, cp, trtbps, chol, fbs, restecg, thalachh,
                                 exng, oldpeak, slp, caa, thall])
        patient_info = mms_scaler.fit_transform(np.expand_dims(patient_info,axis=0))
        y_pred = model.predict(patient_info)
        st.write(heart_attack_chance[y_pred[0]])
        
        if y_pred[0]==1:
            st.warning('You are suspected for having heart attack disease')
        else:
            st.balloons()
            st.success('You\'re free from heart attack suspicion')
            
