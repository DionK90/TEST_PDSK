# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 11:46:24 2022

@author: UPH
"""

import streamlit as st
import pandas as pd
import joblib
st.write("""
# My First Streamlit App
# Penguin Prediction App

This app predicts the **Palmer Penguin** species!

Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header("Input Feature")
input1 = st.sidebar.slider("Input 1:", 0, 10, 5)
sex = st.sidebar.selectbox('Gender',('male','female'))

st.write(f"Input1 oleh user: {input1}")
st.write(f"Input gender: {sex}")

dict_new = {"Input1": input1, "Gender": sex}

df_new = pd.DataFrame(dict_new, index=[0])
st.write(df_new)


###
### Input features untuk Pinguin
###
island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
sex = st.sidebar.selectbox('Sex',('male','female'))
bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
data = {'island': island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': sex}
features = pd.DataFrame(data, index=[0])
st.write(f"Features from user for penguin: ")
st.write(features)


###
### Pemrosesan Data untuk Model Pinguin
###
penguins = pd.read_csv('penguins_cleaned.csv')
df = penguins.copy()
df = df.append(features)
target = 'species'
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
  
st.write(df.iloc[df.shape[0]-1])
features = df[-1:]
features = features.drop("species", axis=1)
# df['species'] = df['species'].apply(target_encode)

### Model Penguin
model = joblib.load("penguins_clf.joblib")
st.write(model.predict(features))
