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
import pickle
import joblib
import sklearn 
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

year = int(my_form.text_input(
     "Enter the new enrollment value"))

gender = my_form.selectbox(
     "Which gender saw an increse in this new session",
     ('Male', 'Female'))

means = my_form.selectbox(
     "What channel of admission was used most during the new session?",
     ("Jamb/Utme", "Direct entry from A'levels", "Transfer from another institution"))

yourself = my_form.selectbox(
     "Did the greater percentage of students who enrolled into RUN make the decisons themselves?",
     ('Yes', 'No'))

location = my_form.selectbox(
     "Did the greater percentage of students enroll into RUN due to the location of the school?",
     ('Yes', 'No'))
st.write('---')


if my_form.form_submit_button('Predict'):
    model = joblib.load('enrol_pred_model.pickle')

    X = pd.DataFrame([[
        gender, year, means, yourself, location]], columns = [
            'Year', 'Gender', 'Means', 'Yourself', 'Location'])

    if year > 2275 : 
        ent = [1]
    else :
        ent = [0]

    # It works now, I guess.... Sleepy head

    X = X.replace([year], ent)
    X = X.replace(['Male', 'Female'], [1, 0])
    X = X.replace(["Jamb/Utme", "Direct entry from A'levels", "Transfer from another institution"], [0, 1, 2])
    X = X.replace(['Yes', 'No'], [0, 1])
    X = X.replace(['Yes', 'No'], [0, 1])

    prediction = model.predict(X)[0]

    if prediction == 1:
        st.markdown('There will be an increase in Enrollment.')
    elif prediction == 0:
        st.markdown('There will be a decrease in Enrollment.')