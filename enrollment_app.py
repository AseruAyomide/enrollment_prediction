#!/usr/bin/env python
# coding: utf-8

# # Prediction of Undergraduate Student Enrollment

# ## Import Libraries

# In[1]:


from cProfile import label
import streamlit as st
import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
#get_ipython().run_line_magic('matplotlib', 'inline')

st.title("""
Predicting Undergraduate Student Enrollment for Redeemer's University

This app predicts undergraduate student enrollment, essentially predicting an increase or decrease in student enrollment 
""")
st.subheader("Enter details below")

#DOCUMENTATION >> st.help(st.form)

#CREATING OUR FORM FIELDS

my_form = st.form(key='form1')

gender = my_form.selectbox("In this new session, which gender increased?",
     ('Male', 'Female'))

means = my_form.selectbox("Which admissions method was most popular during the new session??",
     ("Jamb/Utme", "Direct entry from A'levels", "Transfer from another institution"))

yourself = my_form.selectbox("Does the school have means to personally engage students?",
     ('Yes', 'No'))

friends = my_form.selectbox("Were students urged to promote the school?",
     ('Yes', 'No'))

location = my_form.selectbox("Did the school change location?",
     ('Yes', 'No'))

course = my_form.selectbox("Was there an increase in number of courses offered by the school?",
     ('Yes', 'No'))

fees = my_form.selectbox("Was there an increase of school fees this last session?",
     ('Yes', 'No'))
 
if my_form.form_submit_button('Predict'):
     model = joblib.load('enrol_pred_model.pickle')

     X = pd.DataFrame([[gender, means, yourself, friends, location, course, fees]], columns = ['Gender', 'Means', 'Yourself', 'Friends', 'Location', 'Course', 'Fees'])
     X = X.replace(['Male', 'Female'], [1, 0])
     X = X.replace(["Jamb/Utme", "Direct entry from A'levels", "Transfer from another institution"], [0, 1, 2])
     X = X.replace(['Yes', 'No'], [1, 0])
     X = X.replace(['Yes', 'No'], [1, 0])
     X = X.replace(['Yes', 'No'], [1, 0])
     X = X.replace(['Yes', 'No'], [1, 0])
     X = X.replace(['Yes', 'No'], [1, 0])

     prediction = model.predict(X)[0]

     if prediction == 1:
          st.markdown('There will be an increase in Enrollment.')
     elif prediction == 0:
          st.markdown('There will be a decrease in Enrollment.')