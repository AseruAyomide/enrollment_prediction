#!/usr/bin/env python
# coding: utf-8

# # Prediction of Undergraduate Student Enrollment

# ## Import Libraries

# In[1]:


import streamlit as st
import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import pickle
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# st.title("""
# Predicting Undergraduate Student Enrollment for Redeemer's University

# This app predicts undergraduate student enrollment, essentially predicting an increase or decrease in student enrollment 
# """)
# st.write('---')


data = pd.read_csv("C:\\Users\\ayomi\\Downloads\\Compressed\\h\\Final Year Project Survey.csv")
print(data.shape)
data


# In[3]:


data.describe()


# In[4]:


data.shape


# In[5]:


data.columns


# In[6]:


data_df = pd.DataFrame(data)
data_df


# In[7]:


# st.write('Data Analysis')


# In[8]:


total_rows = len(data_df.index)


# ### Age Range

# In[9]:


Age_less16 = data_df['Age_range'].value_counts()[0]
Age_less16 = int((Age_less16/total_rows)*100)


# In[10]:


Age_16 = data_df['Age_range'].value_counts()[1]
Age_16 = int((Age_16/total_rows)*100)


# In[11]:


Age_17 = data_df['Age_range'].value_counts()[2]
Age_17 = int((Age_17/total_rows)*100)


# In[12]:


Age_18 = data_df['Age_range'].value_counts()[3]
Age_18 = int((Age_18/total_rows)*100)


# In[13]:


Age_over18 = data_df['Age_range'].value_counts()[4]
Age_over18 = int((Age_over18/total_rows)*100)


# In[14]:


Age = pd.DataFrame({'Age range': [0, 1, 2,3,4],
                   'percentage (%)': [Age_less16, Age_16, Age_17, Age_18, Age_over18]})
Age
#st.write("The table below shows the percentage of student who enrolled into Redeemers Uiversity (RUN) at different ages", Age)


# In[15]:


jamb_admin = data_df['Admission_means'].value_counts()[0]
jamb_admin = int((jamb_admin/total_rows)*100)


# In[16]:


direct_admin = data_df['Admission_means'].value_counts()[1]
direct_admin = int((direct_admin/total_rows)*100)


# In[17]:


trans_admin = data_df['Admission_means'].value_counts()[2]
trans_admin = int((trans_admin/total_rows)*100)


# ### Factors

# In[18]:


location_fac = data_df['Factors_Location'].value_counts()[1]
location_fac
loca = int((location_fac/total_rows)*100)


# In[19]:


course_fac = data_df['Factors_Course'].value_counts()[1]
course_fac
cour = int((course_fac/total_rows)*100)


# In[20]:


fees_factor = data_df['Factors_Fees'].value_counts()[1]
fees_factor
fee = int((fees_factor/total_rows)*100)


# In[21]:


rating_fac = data_df['Factors_Rating'].value_counts()[1]
rating_fac
rat = int((rating_fac/total_rows)*100)


# In[22]:


rccg_fac = data_df['Factors_RCCG'].value_counts()[1]
rccg_fac
rccg = int((rccg_fac/total_rows)*100)


# In[23]:


displeasure_fac = data_df['Factors_Displeasure'].value_counts()[1]
displeasure_fac
dis = int((displeasure_fac/total_rows)*100)


# In[24]:


factors = pd.DataFrame({'Factors': ['Location', 'Course of study', 'School fees', 'University rating', 'Association with RCCG', 'Displeasure with previous institution'],
                   'percentage (%)': [loca, cour, fee, rat, rccg, dis],})
factors
#st.write("The table below shows what factors brought about the decision to enroll into Redeemer's University (RUN)", factors)


# ### Influencer

# In[25]:


yourself_inf = data_df['Influencer_Yourself'].value_counts()[1]
yourself_inf
your = int((yourself_inf/total_rows)*100)


# In[26]:


parents_inf = data_df['Influencer_Parents'].value_counts()[1]
parents_inf
p = int((parents_inf/total_rows)*100)


# In[27]:


relatives_inf = data_df['Influencer_Relatives'].value_counts()[1]
relatives_inf
r = int((relatives_inf/total_rows)*100)


# In[28]:


friends_inf = data_df['Influencer_Friends'].value_counts()[1]
friends_inf
f = int((friends_inf/total_rows)*100)


# In[29]:


influencers = pd.DataFrame({'Influence': ['Yourself', 'Parents', 'Relatives', 'Friends'],
                   'percentage (%)': [your, p, r, f],})
influencers
#st.write("The Table below shows the percentage of students who were influenced by various individuals to enroll into Redeemer's University (RUN)", influencers)


# In[30]:


Age = pd.DataFrame({'Age range': [0, 1, 2,3,4],
                   'percentage (%)': [Age_less16, Age_16, Age_17, Age_18, Age_over18]})
Age
#st.write("The table below shows the percentage of student who enrolled into Redeemers U


# In[31]:


rec_count = data_df['Recommendation'].value_counts()[1]
rec_count = int((rec_count/total_rows)*100)
rec_count


# In[32]:


data_fill = data_df.fillna(method="bfill", limit=1)


# In[33]:


data_fill.isnull()


# In[34]:


data_fillna = data_fill.fillna(0)
data_fillna


# In[35]:


data.info()


# In[36]:


data_fill[['FBMS', 'F_ENGR', 'FES', 'F_HUM', 'F_LAW', 'F_MGTSCI', 'FNS', 'F_SOC', 'New_faculty', 'Present_FBMS', 
            'Present_F_HUM', 'Present_FNS','Present_F_SOC']] = data_fillna[['FBMS', 'F_ENGR', 'FES', 
            'F_HUM', 'F_LAW', 'F_MGTSCI', 'FNS', 'F_SOC', 'New_faculty', 'Present_FBMS', 'Present_F_HUM', 
            'Present_FNS','Present_F_SOC']].astype(str)
data_str = data_fill[['FBMS', 'F_ENGR', 'FES', 'F_HUM', 'F_LAW', 'F_MGTSCI', 'FNS', 'F_SOC', 'New_faculty', 
                      'Present_FBMS', 'Present_F_HUM', 'Present_FNS','Present_F_SOC']]


# In[37]:


data_fill[['Gender', 'Admission_year', 'Age_range', 'State_of_residence', 'Admission_means', 'Admission_faculty', 
        'Course_change', 'Present_F_ENGR', 'Present_FES', 'Present_F_LAW', 'Present_F_MGTSCI', 'Scholarship', 
        'Relatives', 'Influencer_Yourself', 'Influencer_Parents', 'Influencer_Relatives', 'Influencer_Friends',
        'Factors_Location', 'Factors_Course', 'Factors_Fees', 'Factors_Rating','Factors_RCCG', 'Factors_Displeasure', 
        'Recommendation', 'enrollment']] = data_fillna[['Gender', 'Admission_year', 'Age_range', 'State_of_residence', 'Admission_means', 
        'Admission_faculty', 'Course_change', 'Present_F_ENGR', 'Present_FES', 'Present_F_LAW', 'Present_F_MGTSCI', 
        'Scholarship', 'Relatives', 'Influencer_Yourself','Influencer_Parents', 'Influencer_Relatives', 'Influencer_Friends',
       'Factors_Location', 'Factors_Course', 'Factors_Fees', 'Factors_Rating','Factors_RCCG', 'Factors_Displeasure', 'Recommendation', 'enrollment']].astype(int)
data_int = data_fill[['Gender', 'Admission_year', 'Age_range', 'State_of_residence', 'Admission_means', 'Admission_faculty', 
        'Course_change', 'Present_F_ENGR', 'Present_FES', 'Present_F_LAW', 'Present_F_MGTSCI', 'Scholarship', 
        'Relatives', 'Influencer_Yourself', 'Influencer_Parents', 'Influencer_Relatives', 'Influencer_Friends',
        'Factors_Location', 'Factors_Course', 'Factors_Fees', 'Factors_Rating','Factors_RCCG', 'Factors_Displeasure', 
        'Recommendation', 'enrollment']]


# In[38]:


data_int.info()


# In[39]:


data_str.info()


# In[40]:


data_drop_str = data_df.drop(['FBMS', 'F_ENGR', 'FES', 'F_HUM', 'F_LAW', 'F_MGTSCI', 'FNS', 'F_SOC', 'New_faculty', 
                      'Present_FBMS', 'Present_FES', 'Present_F_HUM', 'Present_FNS','Present_F_SOC'], axis=1)


# In[41]:


data_drop_str


# In[42]:


data_drop_int = data_df.drop(['Gender', 'Admission_year', 'Age_range', 'State_of_residence', 'Admission_means', 'Admission_faculty', 
        'Course_change', 'Present_F_ENGR', 'Present_FES', 'Present_F_LAW', 'Present_F_MGTSCI', 'Scholarship', 
        'Relatives', 'Influencer_Yourself', 'Influencer_Parents', 'Influencer_Relatives', 'Influencer_Friends',
        'Factors_Location', 'Factors_Course', 'Factors_Fees', 'Factors_Rating','Factors_RCCG', 'Factors_Displeasure', 
        'Recommendation'], axis=1)


# In[43]:


data_drop_int


# In[44]:


#data_concat = pd.concat([data_drop_str, data_drop_int], axis=1)
#data_concat


# In[45]:


data_concat = pd.concat([data_str, data_int], axis=1)
data_concat.info()


# In[46]:


data_concat.columns


# In[47]:


data_concat.describe()


# In[48]:


#LabelEncoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()

data_encode = data_concat.apply(LabelEncoder().fit_transform)
data_encode


# In[49]:


data_encode[data_encode.columns[0]].count()


# In[50]:


X = data_encode.drop(['enrollment'], axis=1)
y = data_encode['enrollment']


# In[51]:


y.shape


# In[52]:


data.columns


# In[53]:


columns = ['FBMS', 'F_ENGR', 'FES', 'F_HUM', 'F_LAW', 'F_MGTSCI', 'FNS', 'F_SOC',
       'New_faculty', 'Present_FBMS', 'Present_F_HUM', 'Present_FNS',
       'Present_F_SOC', 'Gender', 'Admission_year', 'Age_range',
       'State_of_residence', 'Admission_means', 'Admission_faculty',
       'Course_change', 'Present_F_ENGR', 'Present_FES', 'Present_F_LAW',
       'Present_F_MGTSCI', 'Scholarship', 'Relatives', 'Influencer_Yourself',
       'Influencer_Parents', 'Influencer_Relatives', 'Influencer_Friends',
       'Factors_Location', 'Factors_Course', 'Factors_Fees', 'Factors_Rating',
       'Factors_RCCG', 'Factors_Displeasure', 'Recommendation', 'enrollment']


# In[54]:


datum = data_encode.loc[:, columns]
datum


# In[55]:


datum.columns


# In[56]:


features = ['FBMS', 'F_ENGR', 'FES', 'F_HUM', 'F_LAW', 'F_MGTSCI', 'FNS', 'F_SOC',
       'New_faculty', 'Present_FBMS', 'Present_F_HUM', 'Present_FNS',
       'Present_F_SOC', 'Gender', 'Admission_year', 'Age_range',
       'State_of_residence', 'Admission_means', 'Admission_faculty',
       'Course_change', 'Present_F_ENGR', 'Present_FES', 'Present_F_LAW',
       'Present_F_MGTSCI', 'Scholarship', 'Relatives', 'Influencer_Yourself',
       'Influencer_Parents', 'Influencer_Relatives', 'Influencer_Friends',
       'Factors_Location', 'Factors_Course', 'Factors_Fees', 'Factors_Rating',
       'Factors_RCCG', 'Factors_Displeasure', 'Recommendation',]


# In[57]:


temp = datum.drop([
    'Present_F_ENGR', 'Present_FES', 'Present_F_LAW',  'Admission_year', 
    'Present_F_MGTSCI', 'FBMS', 'F_ENGR', 'FES', 'F_HUM', 'F_LAW', 'F_MGTSCI', 'FNS', 'F_SOC',
       'New_faculty', 'Present_FBMS', 'Present_F_HUM', 'Present_FNS',
       'Present_F_SOC', 'State_of_residence'], axis = 1)


# In[58]:


temp.columns


# In[59]:


X= temp.iloc [:, : -1]
y= temp.iloc [:, -1 :]


# In[60]:


y


# In[61]:


X.shape, y.shape


# In[62]:


from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression

lreg = LinearRegression()

sfs1 = sfs(lreg, k_features=7, forward=False, verbose=1, scoring='neg_mean_squared_error')


# In[63]:


sfs1 = sfs1.fit(X, y)


# In[64]:


feat_names = list(sfs1.k_feature_names_)
feat_names


# In[65]:


new_data = data_encode[feat_names]
new_data['enrollment'] = data_encode['enrollment']

new_data.head()


# In[66]:


X= new_data.iloc [:, : -1]
y= new_data.iloc [:, -1 :]


# ### Logistic Regression

# #### Split Data into Train and Test

# In[67]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# In[68]:


#Data Preprocessing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=10)


# In[69]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape 


# #### Trainning the model

# In[70]:


model = LogisticRegression(solver='liblinear', C=8.0, random_state=100)


# In[71]:


model.fit(X_train, y_train)


# #### Predict with test data 

# In[72]:


y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)


# In[73]:


model.classes_


# #### Model Evaluation 

# In[74]:


# Confusion Matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred_test)
cnf_matrix


# In[94]:


print('----- Evaluation on Training Data ----------------------')
# score_train = model.score(X_train, y_train)
# print('Accuracy Score: ', score_train)
# classification report to evaluate the model
print(classification_report(y_train, y_pred_train))
print('--------------------------------------------------------')

print('----- Evaluation on Test Data --------------------------')
# score_test = model.score(X_test, y_test)
# print('Accuracy Score: ', score_test)
# classification report to evaluate the model
print(classification_report(y_test, y_pred_test))
print('--------------------------------------------------------')


# In[96]:


print("Accuracy: {:.2f}" .format(metrics.accuracy_score(y_test, y_pred_test)), '%')
print("Precision: {:.2f}".format(metrics.precision_score(y_test, y_pred_test)), '%')
print("Recall: {:.2f}".format(metrics.recall_score(y_test, y_pred_test)), '%')


# In[77]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[78]:


y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# ### Support Vector Machines

# #### Split Data into Train and Test

# In[79]:


from sklearn.svm import SVC


# In[80]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
#self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, shuffle=False, stratify=None, train_size=training_percent)


# In[81]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape 


# #### Trainning the model

# In[82]:


svclassifier = SVC(kernel='rbf', C = 100)


# In[83]:


svclassifier.fit(X_train, y_train)


# In[98]:


#Save the Model
svc_pickle = open('enrol_pred_model.pickle', 'wb') 
pickle.dump(svclassifier, svc_pickle) 
svc_pickle.close()


# #### Predict with test data 

# In[85]:


y_pred_test = svclassifier.predict(X_test)
y_pred_train = svclassifier.predict(X_train)


# #### Model Evaluation 

# In[86]:


# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred_test))


# In[93]:


print('----- Evaluation on Training Data ----------------------')
#score_train = model.score(X_train, y_train)
#print('Accuracy Score: ', score_train)
# classification report to evaluate the model
print(classification_report(y_train, y_pred_train))
print('--------------------------------------------------------')

print('----- Evaluation on Test Data --------------------------')
#score_test = model.score(X_test, y_test)
#print('Accuracy Score: ', score_test)
# classification report to evaluate the model
print(classification_report(y_test, y_pred_test))
print('--------------------------------------------------------')


# In[88]:


print("Accuracy: {:.2f}" .format(metrics.accuracy_score(y_test, y_pred_test)), '%')
print("Precision: {:.2f}".format(metrics.precision_score(y_test, y_pred_test)), '%')
print("Recall: {:.2f}".format(metrics.recall_score(y_test, y_pred_test)), '%')


# In[89]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred_test)
accuracy = float(cm.diagonal().sum())/len(y_test)


# In[90]:



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[91]:


y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

