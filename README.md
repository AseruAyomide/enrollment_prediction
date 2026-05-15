Student Enrollment Prediction System

A machine learning-powered web application designed to predict whether undergraduate student enrollment in Redeemer’s University is likely to increase or decline based on institutional and student-related factors.

Project Overview

Student enrollment is a major indicator of institutional growth, financial sustainability, and resource planning in higher institutions. Unexpected increases or declines in enrollment can create operational challenges such as overcrowding, underutilized resources, poor infrastructure planning, and reduced student experience.

This project was developed to help institutions make proactive, data-driven decisions by predicting future enrollment trends using machine learning models.

The system analyzes student preferences and institutional factors such as:

School location
Referral/word-of-mouth influence
Admission methods
Scholarship availability
Course availability
Tuition considerations
Student engagement factors
Other enrollment-related variables

After training multiple machine learning models, the best-performing model was deployed as a Streamlit web application where users can input different scenarios and receive enrollment predictions.

Business Problem

Universities often struggle to accurately predict student enrollment trends, which can affect:

Revenue planning
Infrastructure development
Staff recruitment
Hostel allocation
Classroom capacity planning
Academic resource allocation

Without accurate forecasting, institutions may over-invest or under-prepare for future student populations.

This project helps solve that problem by providing a predictive system that enables university management to simulate scenarios and make better strategic decisions.

Dataset

The project used two primary datasets:

1. Student Survey Dataset

Collected using Google Forms from undergraduate students across:

100 level
200 level
300 level
400 level
500 level
Dataset Size:
1,052 rows
38 columns
Key Features:
School location
Scholarship opportunities
Course availability
Referral influence
Tuition affordability
Admission method
Student engagement preferences
2. Historical Enrollment Data

Annual enrollment records collected from Redeemer’s University database from:

2017 – 2021

This data helped provide historical context for model training.

Tools & Technologies Used
Python
Pandas
NumPy
Scikit-learn
Streamlit
Pickle
Google Forms
Jupyter Notebook / Python Scripts
Heroku Deployment
GitHub
Machine Learning Models Used

Two machine learning models were trained and evaluated:

Logistic Regression
Accuracy: 80%
Support Vector Machine (SVM)
Accuracy: 86%

Since SVM performed better, it was selected as the final production model.

Methodology
Data Collection

Survey responses were collected from students using Google Forms.

Data Cleaning & Preprocessing
Removed inconsistencies
Handled missing values
Prepared categorical variables
Structured data for model training
Model Training

Two classification models were trained:

Logistic Regression
Support Vector Machine (SVM)
Model Evaluation

Model performance was evaluated based on prediction accuracy.

Model Deployment

The final SVM model was saved as:

enrol_pred_model.pickle

It was then integrated into a Streamlit application for real-time predictions.

Application Features

The deployed web app allows users to:

✅ Select enrollment-related factors through dropdown menus
✅ Submit institutional scenarios
✅ Receive predictions on whether enrollment may increase or decline
✅ Test multiple admission scenarios

Project Structure
├── Final Year Project Survey.csv
├── Procfile
├── Project_Original_Code.py
├── enrol_pred_model.pickle
├── enrollment_app.py
├── requirements.txt
├── setup.sh

Key Insights

The best-case scenario for increased enrollment included:
Maintaining the school’s current location
Encouraging students to refer others
Improving personal engagement with prospective students
Avoiding tuition increases
Using JAMB/UTME as the primary admission method

These insights can help institutions improve recruitment strategy and enrollment planning.

Future Improvements
Add interactive dashboards and visualisations
Deploy on a more scalable cloud platform
Improve model accuracy with larger datasets
Incorporate real-time admission data
Add explainable AI features to show why predictions are made
Expand the model for use across multiple universities

How to Run Locally
git clone https://github.com/AseruAyomide/enrollment_prediction.git

cd enrollment_prediction

pip install -r requirements.txt

streamlit run enrollment_app.py
